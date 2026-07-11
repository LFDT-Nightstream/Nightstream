import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkArtifact

/-!
Contract: universal semantics of the terminal delayed-link rows in the exact
two-step full-history builder.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLink

def freshOnePair : Nat × Nat := pairs.getD 0 (0, 0)
def freshBitPair (bit : Nat) : Nat × Nat := pairs.getD (bit + 1) (0, 0)

def freshOneCol : Nat := freshOnePair.1
def freshBitCol (bit : Nat) : Nat := (freshBitPair bit).1
def lastXOutBitCol (bit : Nat) : Nat := (freshBitPair bit).2

structure Holds (assignment : Nat → Nat) : Prop where
  affineOne : assignment freshOneCol = 1
  linked : ∀ bit, bit < 256 →
    assignment (freshBitCol bit) = assignment (lastXOutBitCol bit)

private theorem onePair_mem : freshOnePair ∈ pairs := by
  native_decide

private theorem pairs_exact :
    pairs = [freshOnePair] ++ (List.range 256).map freshBitPair := by
  native_decide

private theorem linkPair_mem (bit : Nat) (bitLt : bit < 256) :
    freshBitPair bit ∈ pairs := by
  rw [pairs_exact]
  exact List.mem_append.mpr <| Or.inr <|
    List.mem_map.mpr ⟨bit, List.mem_range.mpr bitLt, rfl⟩

/-- The exact terminal rows fix the affine-one lane and link all 256 public
bits to the recursive producer's canonical `x_out` encoding. -/
theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Holds assignment := by
  have equalities := EqualityPins.rows_sound canonical one satisfies
  refine ⟨(equalities freshOnePair onePair_mem).trans one, ?_⟩
  intro bit bitLt
  exact equalities (freshBitPair bit) (linkPair_mem bit bitLt)

/-- Semantic terminal-link validity satisfies every exact generated row. -/
theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Holds assignment) :
    Satisfies rows assignment := by
  apply EqualityPins.rows_complete canonical one
  intro pair pairMember
  rw [pairs_exact] at pairMember
  simp only [List.mem_append, List.mem_singleton,
    List.mem_map] at pairMember
  rcases pairMember with pairEqual | ⟨bit, bitMember, pairEqual⟩
  · subst pair
    exact holds.affineOne.trans one.symm
  · subst pair
    exact holds.linked bit (List.mem_range.mp bitMember)

theorem satisfies_iff_holds {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies rows assignment ↔ Holds assignment :=
  ⟨sound canonical one, complete canonical one⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound
