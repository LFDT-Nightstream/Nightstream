import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentPage

/-! Axiom audits for the transcripts on the finite wire `Page` that present a path of a game
(`Extract/PresentPage.lean`). The example at the end applies `Tier.exists_block` to a level whose
checks are one round of `Game.sumcheck`, with each hypothesis proved: the game of the level ends
in the result of the run, and the verifier accepts the work slots, the fold challenge and the
nonce of the transcript. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.countDraws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.countDraws_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.nextIndex
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Packs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.packs_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.packs_recv
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Packs.append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Cell.sendable_nat_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Drawable
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Presentable.drawable
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Drawable.bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_cells_from
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_cells
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.prefix_ofFn_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.withWork
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.view_withWork
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.withWork_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_mem_workZeros_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_not_mem_workZeros_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_not_mem_workZeros
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.foldAnswer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.foldAnswer_some
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.foldAnswer_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.foldAnswer_decode
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.challengeAt_of_reply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.decodesAt_iff_of_reply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.foldAnswer_reads
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_weights
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Terminal.fold_of_reply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Prefix.split_of_take
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Prefix.reads_of_split
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Prefix.reads_of_take
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Prefix.ofBlocks
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Prefix.ofBlocks_block
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.game_eq_pure_iff_of_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.exists_block

namespace AkitaPresentPageGuards

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Game (Move)
open NightstreamFPrime.Spec.Compression.Lib (coeffPoly)
open NightstreamFPrime.Spec.Compression.Akita.Extract

local instance : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩

/-- A level whose checks are one round of `Game.sumcheck` of degree 1 over `ZMod 2`, with one
fold coordinate, one stage of the checks and pages of one cell of two elements. -/
noncomputable def roundTier :
    Tier Unit (fun _ => ZMod 2) Unit Unit Unit Unit (Page Unit (ZMod 2) 1 2 0) Unit (ZMod 2)
      (Option (List (ZMod 2) × ZMod 2)) Unit where
  method := ⟨fun _ _ _ => True, fun _ _ => True, fun _ _ _ _ => .inl (), fun _ => True⟩
  claim _ := ()
  block _ := some ()
  decode _ slot := some slot
  read := Page.read
  digits _ := ()
  check _ _ _ := Game.sumcheck 1 (fun _ => 0) 1 [] 0
  passes _ _ := True
  costs := [1 / 2]
  bits := [0]
  foldBits := 0
  start := 0

/-- The round runs on the zero message and the challenge 0 to `some ([0], 0)`. -/
theorem round_runs : (Game.sumcheck 1 (fun _ => (0 : Polynomial (ZMod 2))) 1 [] 0).Runs
    [Move.recv (fun _ : Fin 2 => (0 : ZMod 2)), Move.draw (0 : ZMod 2)] (some ([0], 0)) := by
  have zero : coeffPoly (fun _ : Fin 2 => (0 : ZMod 2)) = 0 := by simp [coeffPoly]
  refine ⟨rfl, ?_⟩
  simp only [cast_eq]
  rw [ite_eq_left (by simp [zero])]
  refine ⟨rfl, ?_⟩
  simp [Game.sumcheck, Game.Runs, zero]

/-- `Tier.exists_block` for `roundTier` and the run `round_runs`: a transcript of two stages on
which the game of the level ends in `some ([0], 0)` and `Tier.Decodes` holds. -/
example : ∃ point : Prefix (Page Unit (ZMod 2) 1 2 0) (Option Unit → ZMod 2) 2,
    roundTier.game () point = .pure (some ([0], 0)) ∧ roundTier.Decodes point := by
  have drawable : Drawable (Unit → ZMod 2)
      (Game.sumcheck 1 (fun _ => (0 : Polynomial (ZMod 2))) 1 [] 0) :=
    (presentable_sumcheck (Coord := Unit) (Slot := ZMod 2) 1 _ 1 [] 0).drawable
  have packs : Packs (ZMod 2) 2 0 1
      [Move.recv (fun _ : Fin 2 => (0 : ZMod 2)), Move.draw (0 : ZMod 2)] 0 :=
    ⟨Nat.one_pos, (Page.sends (E := ZMod 2) (cap := 0) le_rfl).vector 2 _ le_rfl, trivial⟩
  obtain ⟨block, length, -, -, -, reads⟩ := roundTier.exists_block rfl
    ⟨(), fun _ => (fun _ => 0, 0), 0⟩ (fun _ => 0) (c := fun _ => 0) (fun _ => rfl) () round_runs
    drawable packs le_rfl (fun _ => 0) (fun _ => ()) (fun _ => 0)
  let point : Prefix (Page Unit (ZMod 2) 1 2 0) (Option Unit → ZMod 2) 2 :=
    ⟨⟨(), fun _ => (fun _ => 0, 0), 0⟩, block, by rw [length]; exact le_rfl⟩
  obtain ⟨-, -, decodes, -, game⟩ := reads point rfl (by rw [← length]; exact List.take_length)
  refine ⟨point, game, decodes.mpr ⟨?_, fun i inside => ?_, by decide⟩⟩
  · rw [show roundTier.foldBits = 0 from rfl, workZeros_zero]
    exact Finset.mem_univ _
  · have one : i < 1 := inside
    obtain rfl : i = 0 := by omega
    rw [show roundTier.bits.getD 0 0 = 0 from rfl, workZeros_zero]
    exact Finset.mem_univ _

end AkitaPresentPageGuards
