import Nightstream.Implementation.R1CS.FPrimeStateLinkArtifact

/-!
Contract: universal soundness of the exact plain F' state-continuity rows.

Any canonical-residue assignment satisfying the 31 generated rows has exact
wire equality for every state coordinate consumed across adjacent F' steps.
The theorem does not rely on digest equality as authority: every digest lane
and every scalar state coordinate is equated directly.
-/

set_option maxRecDepth 32768

namespace Nightstream.Implementation.R1CS.FPrimeStateLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeStateLink

structure Holds (z : Nat → Nat) : Prop where
  vkFs : ∀ lane, lane < 4 → z (1 + lane) = z (32 + lane)
  header : ∀ lane, lane < 4 → z (5 + lane) = z (36 + lane)
  chunkCount : z 9 = z 40
  stepCount : z 10 = z 41
  initialBoundary : ∀ lane, lane < 4 → z (11 + lane) = z (42 + lane)
  currentBoundary : ∀ lane, lane < 4 → z (15 + lane) = z (46 + lane)
  pc : z 19 = z 50
  semanticState : ∀ lane, lane < 4 → z (20 + lane) = z (51 + lane)
  accumulator : ∀ lane, lane < 4 → z (24 + lane) = z (55 + lane)
  publicTrace : ∀ lane, lane < 4 → z (28 + lane) = z (59 + lane)

private theorem equalityRow_mem {columns : Nat × Nat}
    (member : columns ∈ columnPairs) : equalityRow columns ∈ rows :=
  List.mem_map.mpr ⟨columns, member, rfl⟩

private theorem equality_of_row {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {left right : Nat}
    (holds : RowHolds z (equalityRow (left, right))) :
    z left = z right := by
  have leftLt := hcanon left
  have rightLt := hcanon right
  simp only [equalityRow, RowHolds, lcEval, List.foldl, hone,
    goldilocksP] at holds leftLt rightLt
  omega

private theorem pair_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z)
    {left right : Nat} (member : (left, right) ∈ columnPairs) :
    z left = z right :=
  equality_of_row hcanon hone (hsat _ (equalityRow_mem member))

private theorem digestPair_mem (start lane : Nat) (laneLt : lane < 4)
    (before after : List (Nat × Nat))
    (shape : columnPairs = before ++ digestPairs start ++ after) :
    (start + lane, start + nextOffset + lane) ∈ columnPairs := by
  rw [shape]
  apply List.mem_append_left after
  apply List.mem_append_right before
  exact List.mem_map.mpr ⟨lane, List.mem_range.mpr laneLt, rfl⟩

/-- Every coordinate of adjacent plain F' state bundles is equal. -/
theorem fPrimeStateLink_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z) :
    Holds z := by
  refine {
    vkFs := ?_
    header := ?_
    chunkCount := ?_
    stepCount := ?_
    initialBoundary := ?_
    currentBoundary := ?_
    pc := ?_
    semanticState := ?_
    accumulator := ?_
    publicTrace := ?_
  }
  · intro lane laneLt
    apply pair_sound hcanon hone hsat
    apply digestPair_mem 1 lane laneLt []
      (digestPairs 5 ++ [(9, 40), (10, 41)] ++ digestPairs 11 ++
        digestPairs 15 ++ [(19, 50)] ++ digestPairs 20 ++
        digestPairs 24 ++ digestPairs 28)
    simp [columnPairs]
  · intro lane laneLt
    apply pair_sound hcanon hone hsat
    apply digestPair_mem 5 lane laneLt (digestPairs 1)
      ([(9, 40), (10, 41)] ++ digestPairs 11 ++ digestPairs 15 ++
        [(19, 50)] ++ digestPairs 20 ++ digestPairs 24 ++ digestPairs 28)
    simp [columnPairs]
  · apply pair_sound hcanon hone hsat
    simp [columnPairs]
  · apply pair_sound hcanon hone hsat
    simp [columnPairs]
  · intro lane laneLt
    apply pair_sound hcanon hone hsat
    apply digestPair_mem 11 lane laneLt
      (digestPairs 1 ++ digestPairs 5 ++ [(9, 40), (10, 41)])
      (digestPairs 15 ++ [(19, 50)] ++ digestPairs 20 ++
        digestPairs 24 ++ digestPairs 28)
    simp [columnPairs]
  · intro lane laneLt
    apply pair_sound hcanon hone hsat
    apply digestPair_mem 15 lane laneLt
      (digestPairs 1 ++ digestPairs 5 ++ [(9, 40), (10, 41)] ++
        digestPairs 11)
      ([(19, 50)] ++ digestPairs 20 ++ digestPairs 24 ++ digestPairs 28)
    simp [columnPairs]
  · apply pair_sound hcanon hone hsat
    simp [columnPairs]
  · intro lane laneLt
    apply pair_sound hcanon hone hsat
    apply digestPair_mem 20 lane laneLt
      (digestPairs 1 ++ digestPairs 5 ++ [(9, 40), (10, 41)] ++
        digestPairs 11 ++ digestPairs 15 ++ [(19, 50)])
      (digestPairs 24 ++ digestPairs 28)
    simp [columnPairs]
  · intro lane laneLt
    apply pair_sound hcanon hone hsat
    apply digestPair_mem 24 lane laneLt
      (digestPairs 1 ++ digestPairs 5 ++ [(9, 40), (10, 41)] ++
        digestPairs 11 ++ digestPairs 15 ++ [(19, 50)] ++ digestPairs 20)
      (digestPairs 28)
    simp [columnPairs]
  · intro lane laneLt
    apply pair_sound hcanon hone hsat
    apply digestPair_mem 28 lane laneLt
      (digestPairs 1 ++ digestPairs 5 ++ [(9, 40), (10, 41)] ++
        digestPairs 11 ++ digestPairs 15 ++ [(19, 50)] ++
        digestPairs 20 ++ digestPairs 24)
      []
    simp [columnPairs]

end Nightstream.Implementation.R1CS.FPrimeStateLinkSound
