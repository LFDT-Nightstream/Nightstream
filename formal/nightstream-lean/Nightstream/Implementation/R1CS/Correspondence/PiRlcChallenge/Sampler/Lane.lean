import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk

/-!
Semantic composition of the four candidate leaves in one `Pi_RLC` sampler
lane.

Owns: the protocol -> sampler lane -> four candidate hierarchy; propagation of
the accepted-prefix counter through all four leaves; and the final integer
count equation.

Does not own: transcript/canonical-u64 rows, the other fifteen lanes, the
64-candidate selection tail, production column placement, Rust conformance,
or cost totals.

Emits constraints: no.

Authority boundary: every increment and centered symbol is computed from the
independent `ProductionAlphabet.verifier`. The implementation's accept,
symbol, and cumulative wires appear only on the left side of refinement
equalities.

| Protocol | Phase | Child path | Mathematical obligation | Lean result |
|---|---|---|---|---|
| `Pi_RLC` | sampler/lane | `candidate[0..3]` | each 26-row leaf refines accept/decode/count semantics | `refines.accepted`, `refines.symbols`, `refines.cumulative` |
| `Pi_RLC` | sampler/lane | accepted prefix | each next count is the previous count plus a verifier decision | `refines.cumulative` |
| `Pi_RLC` | sampler/lane | count bound | four Boolean verifier decisions add at most four | `acceptedDelta_le_four` |
| `Pi_RLC` | sampler/lane | lane output | final count is initial count plus four verifier decisions | `refines.finalCount` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Lane

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def part0 : Fin 4 := ⟨0, by decide⟩
def part1 : Fin 4 := ⟨1, by decide⟩
def part2 : Fin 4 := ⟨2, by decide⟩
def part3 : Fin 4 := ⟨3, by decide⟩

def AllSourceBitsBoolean (assignment : Nat -> Nat) : Prop :=
  ∀ part : Fin 4, Chunk.BitsBoolean assignment part.val

def acceptedBit
    (assignment : Nat -> Nat)
    (bits : AllSourceBitsBoolean assignment)
    (part : Fin 4) : Nat :=
  if ProductionAlphabet.verifier.accepts
      (Chunk.candidate assignment part.val (bits part)) then 1 else 0

def expectedSymbol
    (assignment : Nat -> Nat)
    (bits : AllSourceBitsBoolean assignment)
    (part : Fin 4) : Nat :=
  ((ProductionAlphabet.verifier.symbol
      (Chunk.candidate assignment part.val (bits part))).val +
    (goldilocksP - 2)) % goldilocksP

def initialCountCol : Nat := 65
def finalCountCol : Nat := ChunkRows.cumulativeCol 3

def acceptedDelta
    (assignment : Nat -> Nat)
    (bits : AllSourceBitsBoolean assignment) : Nat :=
  acceptedBit assignment bits part0 +
    acceptedBit assignment bits part1 +
    acceptedBit assignment bits part2 +
    acceptedBit assignment bits part3

private theorem part_value_cases (part : Fin 4) :
    part = part0 ∨ part = part1 ∨ part = part2 ∨ part = part3 := by
  have partLt := part.isLt
  have values : part.val = 0 ∨ part.val = 1 ∨
      part.val = 2 ∨ part.val = 3 := by omega
  rcases values with value | value | value | value
  · exact Or.inl (Fin.ext value)
  · exact Or.inr (Or.inl (Fin.ext value))
  · exact Or.inr (Or.inr (Or.inl (Fin.ext value)))
  · exact Or.inr (Or.inr (Or.inr (Fin.ext value)))

private theorem chunkSatisfies
    {assignment : Nat -> Nat}
    (satisfies : Satisfies ChunkRows.rows assignment)
    (part : Fin 4) :
    Satisfies (ChunkRows.chunkRows part.val) assignment := by
  intro row member
  apply satisfies row
  rw [ChunkRows.rows]
  exact List.mem_flatMap.mpr
    ⟨part.val, List.mem_range.mpr part.isLt, member⟩

theorem acceptedBit_le_one
    (assignment : Nat -> Nat)
    (bits : AllSourceBitsBoolean assignment)
    (part : Fin 4) :
    acceptedBit assignment bits part ≤ 1 := by
  unfold acceptedBit
  split <;> simp

theorem acceptedDelta_le_four
    (assignment : Nat -> Nat)
    (bits : AllSourceBitsBoolean assignment) :
    acceptedDelta assignment bits <= 4 := by
  have delta0 := acceptedBit_le_one assignment bits part0
  have delta1 := acceptedBit_le_one assignment bits part1
  have delta2 := acceptedBit_le_one assignment bits part2
  have delta3 := acceptedBit_le_one assignment bits part3
  unfold acceptedDelta
  omega

structure Refines
    (assignment : Nat -> Nat)
    (bits : AllSourceBitsBoolean assignment) : Prop where
  accepted : ∀ part : Fin 4,
    assignment (ChunkRows.acceptCol part.val) =
      acceptedBit assignment bits part
  symbols : ∀ part : Fin 4,
    assignment (ChunkRows.symbolCol part.val) =
      expectedSymbol assignment bits part
  cumulative : ∀ part : Fin 4,
    assignment (ChunkRows.cumulativeCol part.val) =
      assignment (ChunkRows.priorCumulativeCol part.val) +
        acceptedBit assignment bits part
  finalCount : assignment finalCountCol =
    assignment initialCountCol + acceptedDelta assignment bits

theorem refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AllSourceBitsBoolean assignment)
    (initialWithin : assignment initialCountCol + 4 ≤
      ProductionAlphabet.candidateBound)
    (satisfies : Satisfies ChunkRows.rows assignment) :
    Refines assignment bits := by
  have sat0 := chunkSatisfies satisfies part0
  have sat1 := chunkSatisfies satisfies part1
  have sat2 := chunkSatisfies satisfies part2
  have sat3 := chunkSatisfies satisfies part3
  have accept0 := Chunk.acceptanceRows_refine_verifier
    prime canonical one (bits part0) sat0
  have accept1 := Chunk.acceptanceRows_refine_verifier
    prime canonical one (bits part1) sat1
  have accept2 := Chunk.acceptanceRows_refine_verifier
    prime canonical one (bits part2) sat2
  have accept3 := Chunk.acceptanceRows_refine_verifier
    prime canonical one (bits part3) sat3
  have symbol0 := Chunk.symbolRow_refines_verifier
    prime canonical one (bits part0) sat0
  have symbol1 := Chunk.symbolRow_refines_verifier
    prime canonical one (bits part1) sat1
  have symbol2 := Chunk.symbolRow_refines_verifier
    prime canonical one (bits part2) sat2
  have symbol3 := Chunk.symbolRow_refines_verifier
    prime canonical one (bits part3) sat3
  have candidateBound : ProductionAlphabet.candidateBound = 64 := rfl
  rw [candidateBound] at initialWithin
  have initialBound : assignment (ChunkRows.priorCumulativeCol 0) <
      ProductionAlphabet.candidateBound := by
    change assignment initialCountCol < 64
    omega
  have cumulative0Raw := Chunk.cumulativeRow_refines_verifier
    prime canonical one (bits part0) initialBound sat0
  have cumulative0 : assignment (ChunkRows.cumulativeCol 0) =
      assignment initialCountCol + acceptedBit assignment bits part0 := by
    simpa [part0, initialCountCol, acceptedBit,
      ChunkRows.priorCumulativeCol] using cumulative0Raw
  have cumulative0Bound : assignment (ChunkRows.cumulativeCol 0) < 64 := by
    rw [cumulative0]
    have delta := acceptedBit_le_one assignment bits part0
    omega
  have prior1Bound : assignment (ChunkRows.priorCumulativeCol 1) <
      ProductionAlphabet.candidateBound := by
    change assignment (ChunkRows.cumulativeCol 0) < 64
    exact cumulative0Bound
  have cumulative1Raw := Chunk.cumulativeRow_refines_verifier
    prime canonical one (bits part1) prior1Bound sat1
  have cumulative1 : assignment (ChunkRows.cumulativeCol 1) =
      assignment (ChunkRows.cumulativeCol 0) +
        acceptedBit assignment bits part1 := by
    simpa [part1, acceptedBit, ChunkRows.priorCumulativeCol,
      ChunkRows.cumulativeCol, ChunkRows.base] using cumulative1Raw
  have cumulative1Bound : assignment (ChunkRows.cumulativeCol 1) < 64 := by
    rw [cumulative1, cumulative0]
    have delta0 := acceptedBit_le_one assignment bits part0
    have delta1 := acceptedBit_le_one assignment bits part1
    omega
  have prior2Bound : assignment (ChunkRows.priorCumulativeCol 2) <
      ProductionAlphabet.candidateBound := by
    change assignment (ChunkRows.cumulativeCol 1) < 64
    exact cumulative1Bound
  have cumulative2Raw := Chunk.cumulativeRow_refines_verifier
    prime canonical one (bits part2) prior2Bound sat2
  have cumulative2 : assignment (ChunkRows.cumulativeCol 2) =
      assignment (ChunkRows.cumulativeCol 1) +
        acceptedBit assignment bits part2 := by
    simpa [part2, acceptedBit, ChunkRows.priorCumulativeCol,
      ChunkRows.cumulativeCol, ChunkRows.base] using cumulative2Raw
  have cumulative2Bound : assignment (ChunkRows.cumulativeCol 2) < 64 := by
    rw [cumulative2, cumulative1, cumulative0]
    have delta0 := acceptedBit_le_one assignment bits part0
    have delta1 := acceptedBit_le_one assignment bits part1
    have delta2 := acceptedBit_le_one assignment bits part2
    omega
  have prior3Bound : assignment (ChunkRows.priorCumulativeCol 3) <
      ProductionAlphabet.candidateBound := by
    change assignment (ChunkRows.cumulativeCol 2) < 64
    exact cumulative2Bound
  have cumulative3Raw := Chunk.cumulativeRow_refines_verifier
    prime canonical one (bits part3) prior3Bound sat3
  have cumulative3 : assignment (ChunkRows.cumulativeCol 3) =
      assignment (ChunkRows.cumulativeCol 2) +
        acceptedBit assignment bits part3 := by
    simpa [part3, acceptedBit, ChunkRows.priorCumulativeCol,
      ChunkRows.cumulativeCol, ChunkRows.base] using cumulative3Raw
  refine {
    accepted := ?_
    symbols := ?_
    cumulative := ?_
    finalCount := ?_
  }
  · intro part
    rcases part_value_cases part with rfl | rfl | rfl | rfl
    · simpa [acceptedBit] using accept0
    · simpa [acceptedBit] using accept1
    · simpa [acceptedBit] using accept2
    · simpa [acceptedBit] using accept3
  · intro part
    rcases part_value_cases part with rfl | rfl | rfl | rfl
    · simpa [expectedSymbol] using symbol0
    · simpa [expectedSymbol] using symbol1
    · simpa [expectedSymbol] using symbol2
    · simpa [expectedSymbol] using symbol3
  · intro part
    rcases part_value_cases part with rfl | rfl | rfl | rfl
    · simpa [part0, initialCountCol, ChunkRows.priorCumulativeCol] using cumulative0
    · simpa [part1, ChunkRows.priorCumulativeCol,
        ChunkRows.cumulativeCol, ChunkRows.base] using cumulative1
    · simpa [part2, ChunkRows.priorCumulativeCol,
        ChunkRows.cumulativeCol, ChunkRows.base] using cumulative2
    · simpa [part3, ChunkRows.priorCumulativeCol,
        ChunkRows.cumulativeCol, ChunkRows.base] using cumulative3
  · rw [finalCountCol, cumulative3, cumulative2, cumulative1, cumulative0]
    simp only [acceptedDelta]
    omega

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Lane
