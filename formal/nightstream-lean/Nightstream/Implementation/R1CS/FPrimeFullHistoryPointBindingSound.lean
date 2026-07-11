import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePointBindingArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPointBindingArtifact

/-!
Contract: exact semantics and witness completeness of the two NIFS point-binding
owners in the supported full-history artifact. These rows identify both limbs
of the PiDEC parent point with the point emitted by PiCCS; they do not assert
that either surrounding verifier accepted.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound

open Nightstream.Implementation.R1CS

def recursivePair (limb : Nat) : Nat × Nat :=
  FPrimeFullHistoryRecursivePointBinding.pairs.getD limb (0, 0)

def terminalPair (limb : Nat) : Nat × Nat :=
  FPrimeFullHistoryTerminalPointBinding.pairs.getD limb (0, 0)

structure RecursiveHolds (assignment : Nat → Nat) : Prop where
  point : ∀ limb, limb < 2 →
    assignment (recursivePair limb).1 = assignment (recursivePair limb).2

structure TerminalHolds (assignment : Nat → Nat) : Prop where
  point : ∀ limb, limb < 2 →
    assignment (terminalPair limb).1 = assignment (terminalPair limb).2

private theorem recursivePairs_exact :
    FPrimeFullHistoryRecursivePointBinding.pairs =
      (List.range 2).map recursivePair := by
  native_decide

private theorem terminalPairs_exact :
    FPrimeFullHistoryTerminalPointBinding.pairs =
      (List.range 2).map terminalPair := by
  native_decide

private theorem recursivePair_mem (limb : Nat) (limbLt : limb < 2) :
    recursivePair limb ∈ FPrimeFullHistoryRecursivePointBinding.pairs := by
  rw [recursivePairs_exact]
  exact List.mem_map.mpr ⟨limb, List.mem_range.mpr limbLt, rfl⟩

private theorem terminalPair_mem (limb : Nat) (limbLt : limb < 2) :
    terminalPair limb ∈ FPrimeFullHistoryTerminalPointBinding.pairs := by
  rw [terminalPairs_exact]
  exact List.mem_map.mpr ⟨limb, List.mem_range.mpr limbLt, rfl⟩

theorem recursive_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      FPrimeFullHistoryRecursivePointBinding.rows assignment) :
    RecursiveHolds assignment := by
  have equalities := EqualityPins.rows_sound canonical one satisfies
  refine ⟨?_⟩
  intro limb limbLt
  exact equalities (recursivePair limb) (recursivePair_mem limb limbLt)

theorem terminal_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      FPrimeFullHistoryTerminalPointBinding.rows assignment) :
    TerminalHolds assignment := by
  have equalities := EqualityPins.rows_sound canonical one satisfies
  refine ⟨?_⟩
  intro limb limbLt
  exact equalities (terminalPair limb) (terminalPair_mem limb limbLt)

theorem recursive_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RecursiveHolds assignment) :
    Satisfies FPrimeFullHistoryRecursivePointBinding.rows assignment := by
  apply EqualityPins.rows_complete canonical one
  intro pair pairMember
  rw [recursivePairs_exact] at pairMember
  rcases List.mem_map.mp pairMember with ⟨limb, limbMember, rfl⟩
  exact holds.point limb (List.mem_range.mp limbMember)

theorem terminal_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : TerminalHolds assignment) :
    Satisfies FPrimeFullHistoryTerminalPointBinding.rows assignment := by
  apply EqualityPins.rows_complete canonical one
  intro pair pairMember
  rw [terminalPairs_exact] at pairMember
  rcases List.mem_map.mp pairMember with ⟨limb, limbMember, rfl⟩
  exact holds.point limb (List.mem_range.mp limbMember)

theorem recursive_satisfies_iff_holds {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies FPrimeFullHistoryRecursivePointBinding.rows assignment ↔
      RecursiveHolds assignment :=
  ⟨recursive_sound canonical one, recursive_complete canonical one⟩

theorem terminal_satisfies_iff_holds {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies FPrimeFullHistoryTerminalPointBinding.rows assignment ↔
      TerminalHolds assignment :=
  ⟨terminal_sound canonical one, terminal_complete canonical one⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound
