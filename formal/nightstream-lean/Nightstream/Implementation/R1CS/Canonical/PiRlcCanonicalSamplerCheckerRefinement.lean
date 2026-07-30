import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerSound
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker

/-!
Contract: identify the candidate vector reconstructed by the canonical
`Pi_RLC` row program with the exact candidate prefix consumed by the selected
value-level sampler checker.

The load-bearing bridge is binary-source normalization: the four physical
16-bit slices of one canonical-u64 lane are exactly the four little-endian
chunks returned by `PiRlcCanonicalMachine.laneChunk`.  No candidate list,
sampled challenge, or acceptance conclusion is supplied by a caller.

This module owns only the canonical sampler/checker correspondence.  Binding
the initial state and output columns to one concrete NIFS call frame belongs
to the fixed-one lowering layer.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule

private theorem range64_shape : List.range 64 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
     29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
     55, 56, 57, 58, 59, 60, 61, 62, 63] := by
  decide

private theorem finRange16_shape :
    List.finRange PiRlcCanonicalCandidate.sourceBitCount =
      [⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩,
       ⟨3, by decide⟩, ⟨4, by decide⟩, ⟨5, by decide⟩,
       ⟨6, by decide⟩, ⟨7, by decide⟩, ⟨8, by decide⟩,
       ⟨9, by decide⟩, ⟨10, by decide⟩, ⟨11, by decide⟩,
       ⟨12, by decide⟩, ⟨13, by decide⟩, ⟨14, by decide⟩,
       ⟨15, by decide⟩] := by
  decide

/-- One physical 16-bit source slice is the corresponding quotient/modulus
slice of the complete canonical-u64 value. -/
theorem chunkValue_eq_bitsValue_slice
    (assignment : Nat → Nat)
    (layout : CanonicalU64Recipe.Layout)
    (part : Fin 4)
    (bits :
      ∀ index, index < 64 →
        CanonicalU64RecipeSound.bitValue assignment layout index ≤ 1) :
    (List.finRange PiRlcCanonicalCandidate.sourceBitCount).foldl
        (fun value index =>
          value + 2 ^ index.val *
            assignment
              (CanonicalU64Recipe.bitColumn layout
                (part.val * 16 + index.val)))
        0 =
      (CanonicalU64RecipeSound.bitsValue assignment layout /
          (2 ^ (16 * part.val))) %
        chunkModulus := by
  have b0 := bits 0 (by decide)
  have b1 := bits 1 (by decide)
  have b2 := bits 2 (by decide)
  have b3 := bits 3 (by decide)
  have b4 := bits 4 (by decide)
  have b5 := bits 5 (by decide)
  have b6 := bits 6 (by decide)
  have b7 := bits 7 (by decide)
  have b8 := bits 8 (by decide)
  have b9 := bits 9 (by decide)
  have b10 := bits 10 (by decide)
  have b11 := bits 11 (by decide)
  have b12 := bits 12 (by decide)
  have b13 := bits 13 (by decide)
  have b14 := bits 14 (by decide)
  have b15 := bits 15 (by decide)
  have b16 := bits 16 (by decide)
  have b17 := bits 17 (by decide)
  have b18 := bits 18 (by decide)
  have b19 := bits 19 (by decide)
  have b20 := bits 20 (by decide)
  have b21 := bits 21 (by decide)
  have b22 := bits 22 (by decide)
  have b23 := bits 23 (by decide)
  have b24 := bits 24 (by decide)
  have b25 := bits 25 (by decide)
  have b26 := bits 26 (by decide)
  have b27 := bits 27 (by decide)
  have b28 := bits 28 (by decide)
  have b29 := bits 29 (by decide)
  have b30 := bits 30 (by decide)
  have b31 := bits 31 (by decide)
  have b32 := bits 32 (by decide)
  have b33 := bits 33 (by decide)
  have b34 := bits 34 (by decide)
  have b35 := bits 35 (by decide)
  have b36 := bits 36 (by decide)
  have b37 := bits 37 (by decide)
  have b38 := bits 38 (by decide)
  have b39 := bits 39 (by decide)
  have b40 := bits 40 (by decide)
  have b41 := bits 41 (by decide)
  have b42 := bits 42 (by decide)
  have b43 := bits 43 (by decide)
  have b44 := bits 44 (by decide)
  have b45 := bits 45 (by decide)
  have b46 := bits 46 (by decide)
  have b47 := bits 47 (by decide)
  have b48 := bits 48 (by decide)
  have b49 := bits 49 (by decide)
  have b50 := bits 50 (by decide)
  have b51 := bits 51 (by decide)
  have b52 := bits 52 (by decide)
  have b53 := bits 53 (by decide)
  have b54 := bits 54 (by decide)
  have b55 := bits 55 (by decide)
  have b56 := bits 56 (by decide)
  have b57 := bits 57 (by decide)
  have b58 := bits 58 (by decide)
  have b59 := bits 59 (by decide)
  have b60 := bits 60 (by decide)
  have b61 := bits 61 (by decide)
  have b62 := bits 62 (by decide)
  have b63 := bits 63 (by decide)
  simp only [CanonicalU64RecipeSound.bitValue] at b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
  simp only [CanonicalU64RecipeSound.bitValue] at b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31
  simp only [CanonicalU64RecipeSound.bitValue] at b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47
  simp only [CanonicalU64RecipeSound.bitValue] at b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63
  let c0 :=
    (List.finRange PiRlcCanonicalCandidate.sourceBitCount).foldl
      (fun value index =>
        value + 2 ^ index.val *
          assignment (CanonicalU64Recipe.bitColumn layout index.val))
      0
  let c1 :=
    (List.finRange PiRlcCanonicalCandidate.sourceBitCount).foldl
      (fun value index =>
        value + 2 ^ index.val *
          assignment (CanonicalU64Recipe.bitColumn layout (16 + index.val)))
      0
  let c2 :=
    (List.finRange PiRlcCanonicalCandidate.sourceBitCount).foldl
      (fun value index =>
        value + 2 ^ index.val *
          assignment (CanonicalU64Recipe.bitColumn layout (32 + index.val)))
      0
  let c3 :=
    (List.finRange PiRlcCanonicalCandidate.sourceBitCount).foldl
      (fun value index =>
        value + 2 ^ index.val *
          assignment (CanonicalU64Recipe.bitColumn layout (48 + index.val)))
      0
  have c0Lt : c0 < chunkModulus := by
    dsimp [c0]
    rw [finRange16_shape]
    simp only [List.foldl]
    simp
    unfold chunkModulus
    omega
  have c1Lt : c1 < chunkModulus := by
    dsimp [c1]
    rw [finRange16_shape]
    simp only [List.foldl]
    simp
    unfold chunkModulus
    omega
  have c2Lt : c2 < chunkModulus := by
    dsimp [c2]
    rw [finRange16_shape]
    simp only [List.foldl]
    simp
    unfold chunkModulus
    omega
  have c3Lt : c3 < chunkModulus := by
    dsimp [c3]
    rw [finRange16_shape]
    simp only [List.foldl]
    simp
    unfold chunkModulus
    omega
  have wordEq :
      CanonicalU64RecipeSound.bitsValue assignment layout =
        c0 + chunkModulus *
          (c1 + chunkModulus * (c2 + chunkModulus * c3)) := by
    unfold CanonicalU64RecipeSound.bitsValue
      CanonicalU64RecipeSound.bitValue chunkModulus
    dsimp [c0, c1, c2, c3]
    rw [range64_shape, finRange16_shape]
    simp only [List.foldl]
    simp
    omega
  have partCases :
      part.val = 0 ∨ part.val = 1 ∨ part.val = 2 ∨ part.val = 3 := by
    have partLt := part.isLt
    omega
  rcases partCases with partZero | partOne | partTwo | partThree
  · have partEq : part = ⟨0, by decide⟩ := Fin.ext partZero
    rw [partEq]
    simp only [Fin.val_mk, Nat.zero_mul, Nat.mul_zero, Nat.zero_add,
      Nat.pow_zero, Nat.div_one]
    change c0 =
      CanonicalU64RecipeSound.bitsValue assignment layout % chunkModulus
    rw [wordEq, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt c0Lt]
  · have partEq : part = ⟨1, by decide⟩ := Fin.ext partOne
    rw [partEq]
    simp only [Fin.val_mk, Nat.one_mul, Nat.mul_one]
    rw [show 2 ^ 16 = chunkModulus by decide]
    change c1 =
      (CanonicalU64RecipeSound.bitsValue assignment layout / chunkModulus) %
        chunkModulus
    rw [wordEq, Nat.add_mul_div_left c0
      (c1 + chunkModulus * (c2 + chunkModulus * c3)) (by decide)]
    rw [Nat.div_eq_of_lt c0Lt, Nat.zero_add]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt c1Lt]
  · have partEq : part = ⟨2, by decide⟩ := Fin.ext partTwo
    rw [partEq]
    simp only [Fin.val_mk]
    rw [show (2 : Nat) * 16 = 32 by decide,
      show 2 ^ 32 = chunkModulus * chunkModulus by decide]
    change c2 =
      (CanonicalU64RecipeSound.bitsValue assignment layout /
          (chunkModulus * chunkModulus)) %
        chunkModulus
    rw [wordEq]
    rw [← Nat.div_div_eq_div_mul]
    rw [Nat.add_mul_div_left c0
      (c1 + chunkModulus * (c2 + chunkModulus * c3)) (by decide)]
    rw [Nat.div_eq_of_lt c0Lt, Nat.zero_add]
    rw [Nat.add_mul_div_left c1
      (c2 + chunkModulus * c3) (by decide)]
    rw [Nat.div_eq_of_lt c1Lt, Nat.zero_add]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt c2Lt]
  · have partEq : part = ⟨3, by decide⟩ := Fin.ext partThree
    rw [partEq]
    simp only [Fin.val_mk]
    rw [show (3 : Nat) * 16 = 48 by decide,
      show 2 ^ 48 =
        chunkModulus * (chunkModulus * chunkModulus) by decide]
    change c3 =
      (CanonicalU64RecipeSound.bitsValue assignment layout /
          (chunkModulus * (chunkModulus * chunkModulus))) %
        chunkModulus
    rw [wordEq]
    rw [← Nat.div_div_eq_div_mul]
    rw [Nat.add_mul_div_left c0
      (c1 + chunkModulus * (c2 + chunkModulus * c3)) (by decide)]
    rw [Nat.div_eq_of_lt c0Lt, Nat.zero_add]
    rw [← Nat.div_div_eq_div_mul]
    rw [Nat.add_mul_div_left c1
      (c2 + chunkModulus * c3) (by decide)]
    rw [Nat.div_eq_of_lt c1Lt, Nat.zero_add]
    rw [Nat.add_mul_div_left c2 c3 (by decide)]
    rw [Nat.div_eq_of_lt c2Lt, Nat.zero_add]
    rw [Nat.mod_eq_of_lt c3Lt]

/-- The candidate reconstructed from one physical slice is exactly the
little-endian chunk of the value-level digest lane refined by the same
canonical-u64 occurrence. -/
theorem semanticCandidate_eq_digestChunk
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (constants : Poseidon2Schedule.Constants)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar)
    (valid :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (PiRlcCanonicalU64.beforeBlock duplexBase
            (PiRlcCanonicalSymbolicMachine.stateAt
              duplexBase initial coordinate.val)
            coordinate.val (lanePosition candidate))
          (coordinate.val +
            (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val))) :
    PiRlcCanonicalSamplerSound.semanticCandidate
        prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate candidate =
      PiRlcCanonicalMachine.laneChunk
        ((PiRlcCanonicalMachine.digest constants
          (PiRlcCanonicalMachine.appendRawPair constants
            (SymbolicDuplexSemantics.decodedBuilder assignment
              (PiRlcCanonicalU64.beforeBlock duplexBase
                (PiRlcCanonicalSymbolicMachine.stateAt
                  duplexBase initial coordinate.val)
                coordinate.val (lanePosition candidate)))
            1 (coordinate.val +
              (PiRlcCanonicalU64.blockOf
                (lanePosition candidate)).val))).2
          (PiRlcCanonicalU64.laneOf (lanePosition candidate)))
        (address candidate).part := by
  let laneLayout :=
    PiRlcCanonicalU64.laneLayout duplexBase u64Base initial coordinate
      (lanePosition candidate)
  have refined :=
    PiRlcCanonicalU64.lane_refines prime duplexBase u64Base count initial
      canonical constantWire u64Satisfied coordinate (lanePosition candidate)
  have slice :=
    chunkValue_eq_bitsValue_slice assignment laneLayout
      (address candidate).part
      (fun index bounded => refined.bit index bounded)
  have laneEq :=
    PiRlcCanonicalU64.lane_bits_eq_digest
      prime duplexBase u64Base count constants initial canonical constantWire
      u64Satisfied coordinate (lanePosition candidate) valid
  have physicalSlice :
      PiRlcCanonicalCandidateSound.chunkValue assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            candidate) =
        (CanonicalU64RecipeSound.bitsValue assignment laneLayout /
            (2 ^ (16 * (address candidate).part.val))) %
          chunkModulus := by
    simpa [PiRlcCanonicalCandidateSound.chunkValue, candidateLayout,
      sourceBitIndex, laneLayout] using slice
  have finalValue :=
    physicalSlice.trans
      (congrArg
        (fun value =>
          (value / (2 ^ (16 * (address candidate).part.val))) %
            chunkModulus)
        laneEq)
  apply Fin.ext
  simpa [PiRlcCanonicalSamplerSound.semanticCandidate,
    PiRlcCanonicalCandidateSound.candidate,
    PiRlcCanonicalMachine.laneChunk] using finalValue

/-- Candidate address decomposition agrees with the lane occurrence's block
projection. -/
theorem blockOf_lanePosition
    (candidate : Fin candidatesPerScalar) :
    PiRlcCanonicalU64.blockOf (lanePosition candidate) =
      (address candidate).block := by
  apply Fin.ext
  unfold PiRlcCanonicalU64.blockOf lanePosition
  dsimp only
  have laneLt := (address candidate).lane.isLt
  omega

/-- Candidate address decomposition agrees with the lane occurrence's
within-block lane projection. -/
theorem laneOf_lanePosition
    (candidate : Fin candidatesPerScalar) :
    PiRlcCanonicalU64.laneOf (lanePosition candidate) =
      (address candidate).lane := by
  apply Fin.ext
  unfold PiRlcCanonicalU64.laneOf lanePosition
  dsimp only
  have laneLt := (address candidate).lane.isLt
  omega

/-- The symbolic machine's value-state recurrence is definitionally the
state recurrence consumed by the concrete Phi81 sampler checker. -/
theorem valueStateAt_eq_samplerStateAt
    (constants : Poseidon2Schedule.Constants)
    (initial : Poseidon2Duplex.State) :
    ∀ coordinate,
      PiRlcCanonicalSymbolicMachine.valueStateAt
          constants initial coordinate =
        stateAt
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
            (PiRlcCanonicalMachine.machine constants))
          initial coordinate
  | 0 => rfl
  | coordinate + 1 => by
      rw [PiRlcCanonicalSymbolicMachine.valueStateAt, stateAt]
      rw [valueStateAt_eq_samplerStateAt constants initial coordinate]
      rfl

private theorem stateBeforeBlock_extends_of_le
    (base : Nat) (entered : SymbolicDuplex.Builder) (seed : Nat)
    {start finish : Nat} (within : start ≤ finish) :
    SymbolicDuplexSemantics.Extends
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
        base entered seed start)
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
        base entered seed finish) := by
  obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le within
  induction extra with
  | zero =>
      simpa using
        (SymbolicDuplexSemantics.Extends.refl
          (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
            base entered seed start))
  | succ extra inductionHypothesis =>
      have extensionPrefix :
          SymbolicDuplexSemantics.Extends
            (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
              base entered seed start)
            (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
              base entered seed (start + extra)) :=
        inductionHypothesis (by omega)
      have next :
          PiRlcCanonicalSymbolicMachine.stateBeforeBlock
              base entered seed (start + (extra + 1)) =
            PiRlcCanonicalSymbolicMachine.digestBlock base
              (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
                base entered seed (start + extra))
              (seed + (start + extra)) := by
        rw [show start + (extra + 1) = (start + extra) + 1 by omega]
        rfl
      rw [next]
      exact extensionPrefix.trans
        (PiRlcCanonicalSymbolicMachine.digestBlock_extends base
          (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
            base entered seed (start + extra))
          (seed + (start + extra)))

private theorem stateAt_extends_of_le
    (base : Nat) (initial : SymbolicDuplex.Builder)
    {start finish : Nat} (within : start ≤ finish) :
    SymbolicDuplexSemantics.Extends
      (PiRlcCanonicalSymbolicMachine.stateAt base initial start)
      (PiRlcCanonicalSymbolicMachine.stateAt base initial finish) := by
  obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le within
  induction extra with
  | zero =>
      simpa using
        (SymbolicDuplexSemantics.Extends.refl
          (PiRlcCanonicalSymbolicMachine.stateAt base initial start))
  | succ extra inductionHypothesis =>
      have extensionPrefix :
          SymbolicDuplexSemantics.Extends
            (PiRlcCanonicalSymbolicMachine.stateAt base initial start)
            (PiRlcCanonicalSymbolicMachine.stateAt
              base initial (start + extra)) :=
        inductionHypothesis (by omega)
      have next :
          PiRlcCanonicalSymbolicMachine.stateAt
              base initial (start + (extra + 1)) =
            PiRlcCanonicalSymbolicMachine.scalarBuilder base
              (PiRlcCanonicalSymbolicMachine.stateAt
                base initial (start + extra))
              (start + extra) := by
        rw [show start + (extra + 1) = (start + extra) + 1 by omega]
        rfl
      rw [next]
      exact extensionPrefix.trans
        (PiRlcCanonicalSymbolicMachine.scalarBuilder_extends base
          (PiRlcCanonicalSymbolicMachine.stateAt
            base initial (start + extra))
          (start + extra))

/-- Every physical candidate reconstructed from a globally valid symbolic
batch is the candidate at the same position of the executable checker's
verifier-owned source stream. -/
theorem semanticCandidate_eq_checkerStream
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (constants : Poseidon2Schedule.Constants)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (validBatch :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial count))
    (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) :
    PiRlcCanonicalSamplerSound.semanticCandidate
        prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate candidate =
      (sourceAt
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
          (PiRlcCanonicalMachine.machine constants))
        (SymbolicDuplexSemantics.decodedBuilder assignment initial)
        coordinate.val).stream candidate.val := by
  have blockSuccLe :
      (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val + 1 ≤
        digestRounds := by
    have blockLt :=
      (PiRlcCanonicalU64.blockOf (lanePosition candidate)).isLt
    simp only [digestRounds] at blockLt ⊢
    omega
  have laneToScalar :
      SymbolicDuplexSemantics.Extends
        (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (PiRlcCanonicalU64.beforeBlock duplexBase
            (PiRlcCanonicalSymbolicMachine.stateAt
              duplexBase initial coordinate.val)
            coordinate.val (lanePosition candidate))
          (coordinate.val +
            (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val))
        (PiRlcCanonicalSymbolicMachine.scalarBuilder duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val) := by
    change SymbolicDuplexSemantics.Extends
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock duplexBase
        (PiRlcCanonicalSymbolicMachine.enterScalar duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val)
        coordinate.val
        ((PiRlcCanonicalU64.blockOf (lanePosition candidate)).val + 1))
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock duplexBase
        (PiRlcCanonicalSymbolicMachine.enterScalar duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val)
        coordinate.val digestRounds)
    exact stateBeforeBlock_extends_of_le duplexBase
      (PiRlcCanonicalSymbolicMachine.enterScalar duplexBase
        (PiRlcCanonicalSymbolicMachine.stateAt
          duplexBase initial coordinate.val)
        coordinate.val)
      coordinate.val blockSuccLe
  have scalarToBatch :
      SymbolicDuplexSemantics.Extends
        (PiRlcCanonicalSymbolicMachine.scalarBuilder duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val)
        (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial count) := by
    change SymbolicDuplexSemantics.Extends
      (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial
        (coordinate.val + 1))
      (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial count)
    exact stateAt_extends_of_le duplexBase initial
      (Nat.succ_le_of_lt coordinate.isLt)
  have validLane :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (PiRlcCanonicalU64.beforeBlock duplexBase
            (PiRlcCanonicalSymbolicMachine.stateAt
              duplexBase initial coordinate.val)
            coordinate.val (lanePosition candidate))
          (coordinate.val +
            (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val)) :=
    validBatch.of_extends (laneToScalar.trans scalarToBatch)
  have validBefore :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalU64.beforeBlock duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val (lanePosition candidate)) :=
    validLane.of_extends
      (PiRlcCanonicalSymbolicMachine.digestBlock_extends duplexBase
        (PiRlcCanonicalU64.beforeBlock duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val (lanePosition candidate))
        (coordinate.val +
          (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val))
  have validEntered :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.enterScalar duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val) :=
    validBefore.of_extends
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock_extends duplexBase
        (PiRlcCanonicalSymbolicMachine.enterScalar duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initial coordinate.val)
          coordinate.val)
        coordinate.val
        (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val)
  have validState :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.stateAt
          duplexBase initial coordinate.val) :=
    validEntered.of_extends
      (PiRlcCanonicalSymbolicMachine.enterScalar_extends duplexBase
        (PiRlcCanonicalSymbolicMachine.stateAt
          duplexBase initial coordinate.val)
        coordinate.val)
  have decodedState :=
    PiRlcCanonicalSymbolicMachine.decoded_stateAt duplexBase constants
      assignment constantWire initial coordinate.val validState
  have decodedEntered :=
    PiRlcCanonicalSymbolicMachine.decoded_enterScalar duplexBase constants
      assignment constantWire
      (PiRlcCanonicalSymbolicMachine.stateAt
        duplexBase initial coordinate.val)
      coordinate.val validEntered
  have decodedBefore :=
    PiRlcCanonicalSymbolicMachine.decoded_stateBeforeBlock duplexBase constants
      assignment constantWire
      (PiRlcCanonicalSymbolicMachine.enterScalar duplexBase
        (PiRlcCanonicalSymbolicMachine.stateAt
          duplexBase initial coordinate.val)
        coordinate.val)
      coordinate.val
      (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val
      validBefore
  have checkerState :
      SymbolicDuplexSemantics.decodedBuilder assignment
          (PiRlcCanonicalU64.beforeBlock duplexBase
            (PiRlcCanonicalSymbolicMachine.stateAt
              duplexBase initial coordinate.val)
            coordinate.val (lanePosition candidate)) =
        ProductionSchedule.stateBeforeBlock
          (PiRlcCanonicalMachine.machine constants)
          ((PiRlcCanonicalMachine.machine constants).enterScalar
            (stateAt
              (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
                (PiRlcCanonicalMachine.machine constants))
              (SymbolicDuplexSemantics.decodedBuilder assignment initial)
              coordinate.val)
            coordinate.val)
          coordinate.val
          (PiRlcCanonicalU64.blockOf (lanePosition candidate)).val := by
    rw [PiRlcCanonicalU64.beforeBlock]
    rw [decodedBefore]
    congr 1
    rw [decodedEntered]
    rw [decodedState]
    rw [valueStateAt_eq_samplerStateAt constants
      (SymbolicDuplexSemantics.decodedBuilder assignment initial)
      coordinate.val]
    rfl
  rw [semanticCandidate_eq_digestChunk prime duplexBase u64Base
    candidateBase count constants initial canonical constantWire
    u64Satisfied coordinate candidate validLane]
  rw [checkerState]
  unfold sourceAt
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
    ProductionSchedule.specification ProductionSchedule.source
    ProductionSchedule.candidateStream ProductionSchedule.chunksAt
  dsimp only
  rw [show candidate.val / chunksPerDigest =
      (address candidate).block.val by
    simp only [chunksPerDigest]
    have recomposes := address_recomposes candidate
    have laneLt := (address candidate).lane.isLt
    have partLt := (address candidate).part.isLt
    omega]
  have withinBlock :
      candidate.val % chunksPerDigest =
        (address candidate).lane.val * 4 +
          (address candidate).part.val := by
    simp only [chunksPerDigest]
    have recomposes := address_recomposes candidate
    have laneLt := (address candidate).lane.isLt
    have partLt := (address candidate).part.isLt
    omega
  rw [show (⟨candidate.val % chunksPerDigest,
      Nat.mod_lt _ (by decide)⟩ : Fin chunksPerDigest) =
        ⟨(address candidate).lane.val * 4 +
          (address candidate).part.val, by
            have laneLt := (address candidate).lane.isLt
            have partLt := (address candidate).part.isLt
            simp only [chunksPerDigest]
            omega⟩ by
      apply Fin.ext
      exact withinBlock]
  rw [blockOf_lanePosition candidate, laneOf_lanePosition candidate]
  change PiRlcCanonicalMachine.laneChunk
      ((PiRlcCanonicalMachine.digest constants
        (PiRlcCanonicalMachine.appendRawPair constants
          (ProductionSchedule.stateBeforeBlock
            (PiRlcCanonicalMachine.machine constants)
            ((PiRlcCanonicalMachine.machine constants).enterScalar
              (stateAt
                (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
                  (PiRlcCanonicalMachine.machine constants))
                (SymbolicDuplexSemantics.decodedBuilder assignment initial)
                coordinate.val)
              coordinate.val)
            coordinate.val (address candidate).block.val)
          1 (coordinate.val + (address candidate).block.val))).2
        (address candidate).lane)
      (address candidate).part =
    PiRlcCanonicalMachine.digestChunks
      ((PiRlcCanonicalMachine.digest constants
        (PiRlcCanonicalMachine.appendRawPair constants
          (ProductionSchedule.stateBeforeBlock
            (PiRlcCanonicalMachine.machine constants)
            ((PiRlcCanonicalMachine.machine constants).enterScalar
              (stateAt
                (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
                  (PiRlcCanonicalMachine.machine constants))
                (SymbolicDuplexSemantics.decodedBuilder assignment initial)
                coordinate.val)
              coordinate.val)
            coordinate.val (address candidate).block.val)
          1 (coordinate.val + (address candidate).block.val))).2)
      ⟨(address candidate).lane.val * 4 +
        (address candidate).part.val, by
          have laneLt := (address candidate).lane.isLt
          have partLt := (address candidate).part.isLt
          simp only [chunksPerDigest]
          omega⟩
  rw [PiRlcCanonicalMachine.digestChunks_lane_part]

/-- The complete 64-element physical candidate vector is definitionally the
same bounded prefix inspected by the executable checker. -/
theorem semanticCandidates_eq_candidatePrefix
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (constants : Poseidon2Schedule.Constants)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (validBatch :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial count))
    (coordinate : Fin count) :
    PiRlcCanonicalSamplerSound.semanticCandidates
        prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate =
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.candidatePrefix
        (PiRlcCanonicalMachine.machine constants)
        (SymbolicDuplexSemantics.decodedBuilder assignment initial)
        coordinate.val := by
  apply List.ext_getElem
  · rw [PiRlcCanonicalSamplerSound.semanticCandidates_length]
    simp [
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.candidatePrefix,
      FirstAccepted.streamPrefix, candidateBound, candidatesPerScalar]
  · intro index leftBound _rightBound
    have indexLt : index < candidatesPerScalar := by
      rw [PiRlcCanonicalSamplerSound.semanticCandidates_length] at leftBound
      exact leftBound
    let candidate : Fin candidatesPerScalar := ⟨index, indexLt⟩
    change
      (List.ofFn fun candidate : Fin candidatesPerScalar =>
        PiRlcCanonicalSamplerSound.semanticCandidate
          prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate candidate)[index] =
        ((List.range candidateBound).map
          (sourceAt
            (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
              (PiRlcCanonicalMachine.machine constants))
            (SymbolicDuplexSemantics.decodedBuilder assignment initial)
            coordinate.val).stream)[index]
    rw [List.getElem_ofFn, List.getElem_map, List.getElem_range]
    exact semanticCandidate_eq_checkerStream prime duplexBase u64Base
      candidateBase count constants initial canonical constantWire
      u64Satisfied validBatch coordinate candidate

/-- The row-level `Enough` theorem and exact stream equality produce the
checker's successful bounded-sampler result, with no termination premise. -/
theorem boundedSample_eq_semanticOutput
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (constants : Poseidon2Schedule.Constants)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (selectorSatisfied :
      Satisfies
        (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
          selectorBase count initial)
        assignment)
    (validBatch :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial count))
    (coordinate : Fin count) :
    FirstAccepted.boundedSample verifier coefficientCount
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.candidatePrefix
          (PiRlcCanonicalMachine.machine constants)
          (SymbolicDuplexSemantics.decodedBuilder assignment initial)
          coordinate.val) =
      some
        (PiRlcCanonicalSamplerSound.semanticOutput
          prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate) := by
  apply FirstAccepted.boundedSample_eq_some_iff.mpr
  constructor
  · rw [← semanticCandidates_eq_candidatePrefix prime duplexBase u64Base
      candidateBase count constants initial canonical constantWire
      u64Satisfied validBatch coordinate]
    simpa [coefficientCount, PiRlcCanonicalSelector.outputCount] using
      (PiRlcCanonicalSamplerSound.enough prime duplexBase u64Base
        candidateBase selectorBase count initial canonical constantWire
        u64Satisfied candidateSatisfied selectorSatisfied coordinate)
  · unfold PiRlcCanonicalSamplerSound.semanticOutput
    rw [semanticCandidates_eq_candidatePrefix prime duplexBase u64Base
      candidateBase count constants initial canonical constantWire
      u64Satisfied validBatch coordinate]
    rfl

/-- The exact RingF value assembled from the physical first-accepted output. -/
def semanticChallenge
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) :
    Nightstream.SuperNeo.Concrete.RingF :=
  Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedScalar
    (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.scalarOfList
      (PiRlcCanonicalSamplerSound.semanticOutput
        prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate))

/-- Executing the selected checker on the decoded symbolic initial state
returns exactly the RingF challenge assembled from the physical sampler. -/
theorem sampleChallenge?_eq_some_semanticChallenge
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (constants : Poseidon2Schedule.Constants)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (selectorSatisfied :
      Satisfies
        (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
          selectorBase count initial)
        assignment)
    (validBatch :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial count))
    (coordinate : Fin count) :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
        (PiRlcCanonicalMachine.machine constants)
        (SymbolicDuplexSemantics.decodedBuilder assignment initial)
        coordinate.val =
      some
        (semanticChallenge prime duplexBase u64Base candidateBase count
          initial canonical constantWire u64Satisfied coordinate) := by
  unfold
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
    semanticChallenge
  rw [boundedSample_eq_semanticOutput prime duplexBase u64Base candidateBase
    selectorBase count constants initial canonical constantWire u64Satisfied
    candidateSatisfied selectorSatisfied validBatch coordinate]
  rfl

/-- Reinterpret one physical selector position as the definitionally equal
Phi81 ring-coordinate position. -/
def outputRingPosition
    (position : Fin PiRlcCanonicalSelector.outputCount) :
    Fin Nightstream.SuperNeo.Concrete.ringDegree :=
  ⟨position.val, by
    have bounded := position.isLt
    simpa [PiRlcCanonicalSelector.outputCount,
      Nightstream.SuperNeo.Concrete.ringDegree] using bounded⟩

/-- Every selector output column is the matching coordinate of the exact
RingF value returned by `sampleChallenge?`. -/
theorem semanticChallenge_coordinate_eq_outputColumn
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (selectorSatisfied :
      Satisfies
        (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
          selectorBase count initial)
        assignment)
    (coordinate : Fin count)
    (position : Fin PiRlcCanonicalSelector.outputCount) :
    (semanticChallenge prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate
        (outputRingPosition position)).val =
      assignment
        (PiRlcCanonicalSelector.outputColumn
          selectorBase coordinate position) := by
  have enough :=
    PiRlcCanonicalSamplerSound.enough prime duplexBase u64Base candidateBase
      selectorBase count initial canonical constantWire u64Satisfied
      candidateSatisfied selectorSatisfied coordinate
  have outputLength :
      (PiRlcCanonicalSamplerSound.semanticOutput
        prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate).length =
        PiRlcCanonicalSelector.outputCount :=
    FirstAccepted.firstAccepted_length_of_enough enough
  have inBounds :
      position.val <
        (PiRlcCanonicalSamplerSound.semanticOutput
          prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate).length := by
    rw [outputLength]
    exact position.isLt
  have outputEq :=
    PiRlcCanonicalSamplerSound.output_getElem?_eq prime duplexBase u64Base
      candidateBase selectorBase count initial canonical constantWire
      u64Satisfied candidateSatisfied selectorSatisfied coordinate position
  rw [List.getElem?_eq_getElem inBounds] at outputEq
  simp only [Option.map_some, Option.some.injEq] at outputEq
  unfold semanticChallenge
    Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedScalar
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.scalarOfList
  change
    (Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedCoefficient
      ((PiRlcCanonicalSamplerSound.semanticOutput
        prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate).getD
          position.val
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.defaultCoefficient)).val =
      assignment
        (PiRlcCanonicalSelector.outputColumn
          selectorBase coordinate position)
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inBounds]
  exact outputEq

/-- Satisfaction of the complete fixed-active sampler program is sufficient
to execute the selected checker successfully at every one of its fifteen
coordinates. -/
theorem samplerRows_sampleChallenge?_eq_some
    (prime : EuclidPrime goldilocksP)
    (duplexBase : Nat)
    (constants : Poseidon2Schedule.Constants)
    (lanes : Poseidon2Core.State)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (PiRlcCanonicalSamplerProgram.rows duplexBase constants lanes)
        assignment)
    (coordinate :
      Fin PiRlcCanonicalSamplerProgram.coordinateCount) :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
        (PiRlcCanonicalMachine.machine constants)
        (SymbolicDuplexSemantics.decodedBuilder assignment
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
        coordinate.val =
      some
        (semanticChallenge prime duplexBase
          (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
          canonical constantWire
          (PiRlcCanonicalSamplerProgram.u64Rows_satisfied duplexBase constants
            lanes assignment satisfied)
          coordinate) := by
  let initial := PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
  let u64Satisfied :=
    PiRlcCanonicalSamplerProgram.u64Rows_satisfied duplexBase constants
      lanes assignment satisfied
  let candidateSatisfied :=
    PiRlcCanonicalSamplerProgram.candidateRows_satisfied duplexBase constants
      lanes assignment satisfied
  let selectorSatisfied :=
    PiRlcCanonicalSamplerProgram.selectorRows_satisfied duplexBase constants
      lanes assignment satisfied
  have validBatch :
      SymbolicDuplexSemantics.Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial
          PiRlcCanonicalSamplerProgram.coordinateCount) := by
    apply SymbolicDuplexSemantics.valid_of_satisfied duplexBase constants
      (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initial
        PiRlcCanonicalSamplerProgram.coordinateCount)
      assignment canonical constantWire
    simpa [PiRlcCanonicalSamplerProgram.transcriptRows,
      PiRlcCanonicalSymbolicMachineHonest.fixedBuilder,
      PiRlcCanonicalSamplerProgram.coordinateCount, initial] using
      (PiRlcCanonicalSamplerProgram.transcriptRows_satisfied
        duplexBase constants lanes assignment satisfied)
  exact sampleChallenge?_eq_some_semanticChallenge prime duplexBase
    (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
    (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
    (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
    PiRlcCanonicalSamplerProgram.coordinateCount constants initial canonical
    constantWire u64Satisfied candidateSatisfied selectorSatisfied validBatch
    coordinate

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement
