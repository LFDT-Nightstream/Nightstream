import CompressionLean.Shrink.Soundness

/-!
Owns the error of the shrink verifier game and the main theorem
(`shrink_sound`): against every adaptive prover, the verifier accepts while
no candidate satisfies its program's CCS with probability at most
`shrinkError`.

The count is per candidate: `τ` (at most one bad value per coordinate,
`foldBad`), the outer rounds (degree 8 per round), the batching (degree 7)
and the inner rounds (degree 2 per round).
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

/-- The shrink bad-challenge count over `L` candidates, `m` row variables
and `n` cell variables: `τ`, the outer rounds, the batching, the inner
rounds. -/
def shrinkCount (m n L : ℕ) : ℕ := m * L + m * (L * 8) + L * 7 + (n + 1) * (L * 2)

/-- The shrink error over a challenge field of `size` elements. -/
noncomputable def shrinkError (m n L size : ℕ) : ℝ := (shrinkCount m n L : ℝ) / size

theorem programGame_cost {I : Type} (i : I) (C : Ccs F) (shapeOk : Prop) (matrices : List K → List K → K → K)
    (L : ℕ) (nodes7 : NodesDistinct K 7) (nodes1 : NodesDistinct K 1) :
    (programGame (K := K) i C shapeOk matrices L).CostLe (shrinkError C.rowVars C.cellVars L (Fintype.card K)) := by
  intro cands
  have count : (Finset.univ : Finset (Fin cands.1.length)).card ≤ L := by simpa using cands.2
  have positive : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
  have tauCard : ∀ point : List K, (tauBad C cands.1 point).card ≤ L := fun point =>
    (card_biUnion_le_mul _ _ 1 fun k _ => foldBad_card _ _ _).trans (by simpa using count)
  refine (Game.costLe_bind (Game.drawPoint_cost (tauBad C cands.1) L tauCard C.rowVars []) fun tau =>
    Game.costLe_bind
      (δ := ((L * 7 : ℕ) : ℝ) / Fintype.card K + ((C.cellVars + 1 : ℕ) : ℝ) *
        ((((Finset.univ : Finset (Fin cands.1.length)).card * (1 + 1) + 0 : ℕ) : ℝ) / (Fintype.card K : ℝ)))
      (Game.sumcheck_cost Finset.univ 7 nodes7 _ (fun _ => ∅) C.rowVars 0
        (fun k _ => (combRound_truth C.rowVars 7 (outerTables C.entries (C.z (cands.1.get k)) tau) outerComb
          outerComb_degree).degree)
        (fun _ => by simp) C.rowVars [] 0 (by simp)) fun outer => ?_).mono ?_
  · intro values
    dsimp only
    split_ifs
    · refine ⟨((C.cellVars + 1 : ℕ) : ℝ) *
        ((((Finset.univ : Finset (Fin cands.1.length)).card * (1 + 1) + 0 : ℕ) : ℝ) / (Fintype.card K : ℝ)),
        ?_, fun rho => ?_⟩
      · have card : (rhoBad C cands.1 outer.1 values).card ≤ L * 7 :=
          (card_biUnion_le_mul _ _ 7 fun k _ => zeros_card_le_of_natDegree_le (batchPoly_natDegree _ _)).trans
            (Nat.mul_le_mul_right 7 count)
        have := Game.density_le _ _ card
        linarith
      · refine (Game.costLe_bind (δ := 0) (Game.sumcheck_cost Finset.univ 1 nodes1 _ (fun _ => ∅) (C.cellVars + 1) 0
          (fun k _ => (combRound_truth (C.cellVars + 1) 1
            (innerTables C.entries (C.z (cands.1.get k)) outer.1 rho) productComb productComb_degree).degree)
          (fun _ => by simp) (C.cellVars + 1) [] _ (by simp)) fun inner => ?_).mono (le_of_eq (add_zero _))
        intro opened
        dsimp only
        split_ifs <;> exact le_rfl
    · show (0 : ℝ) ≤ _
      positivity
  · have c : ((Finset.univ : Finset (Fin cands.1.length)).card : ℝ) ≤ L := by exact_mod_cast count
    have m0 : (0 : ℝ) ≤ C.rowVars := Nat.cast_nonneg _
    have n0 : (0 : ℝ) ≤ (C.cellVars : ℝ) + 1 := by positivity
    have numerator : (C.rowVars : ℝ) * L +
        C.rowVars * (((Finset.univ : Finset (Fin cands.1.length)).card : ℝ) * 8) + L * 7 +
        ((C.cellVars : ℝ) + 1) * (((Finset.univ : Finset (Fin cands.1.length)).card : ℝ) * 2) ≤
        C.rowVars * L + C.rowVars * (L * 8) + L * 7 + ((C.cellVars : ℝ) + 1) * (L * 2) := by
      nlinarith [mul_le_mul_of_nonneg_left c m0, mul_le_mul_of_nonneg_left c n0]
    calc _ = ((C.rowVars : ℝ) * L +
            C.rowVars * (((Finset.univ : Finset (Fin cands.1.length)).card : ℝ) * 8) + L * 7 +
            ((C.cellVars : ℝ) + 1) * (((Finset.univ : Finset (Fin cands.1.length)).card : ℝ) * 2)) /
            Fintype.card K := by
          push_cast
          ring
      _ ≤ ((C.rowVars : ℝ) * L + C.rowVars * (L * 8) + L * 7 + ((C.cellVars : ℝ) + 1) * (L * 2)) /
            Fintype.card K := div_le_div_of_nonneg_right numerator positive.le
      _ = shrinkError C.rowVars C.cellVars L (Fintype.card K) := by
          unfold shrinkError shrinkCount
          push_cast
          ring

theorem shrinkGame_cost {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F) (L : ℕ)
    (nodes7 : NodesDistinct K 7) (nodes1 : NodesDistinct K 1) :
    (shrinkGame (K := K) T s prog L).CostLe
      (shrinkError (Layout.rowVars T s) (Layout.cellVars T s) L (Fintype.card K)) :=
  fun i => programGame_cost i ((prog i).ccs T s) ((prog i).counts = s) (verifierMatrices T s (prog i)) L
    nodes7 nodes1

/-- **Shrink soundness with an idealized WHIR opening.** Against every
adaptive prover, the shrink verifier accepts while no committed candidate
`w` satisfies the CCS of the named program, `z = (w, 1, x)` with its
statement `x`, with probability at most `shrinkError`. -/
theorem shrink_sound {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F) (L : ℕ)
    (wf : T.WellFormed) (valid : ∀ i, (prog i).Valid T)
    (cast : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b)
    (prover : Game.Strategy (Option (Transcript I F K))) :
    Game.prob prover (shrinkGame T s prog L)
        (fun r => ∃ tr, r = some tr ∧ ¬ ∃ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w) ≤
      shrinkError (Layout.rowVars T s) (Layout.cellVars T s) L (Fintype.card K) := by
  rw [Game.prob_congr _ _ (E' := fun r =>
    ¬ ∀ tr, r = some tr → ∃ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w)]
  · exact Game.prob_not_le _ _ _ _
      (shrinkGame_avoids T s prog L wf valid (shrinkNodes cast 7 le_rfl) (shrinkNodes cast 1 (by norm_num)))
      (shrinkGame_cost T s prog L (shrinkNodes cast 7 le_rfl) (shrinkNodes cast 1 (by norm_num)))
  · intro r
    constructor
    · rintro ⟨tr, rfl, none⟩ all
      exact none (all tr rfl)
    · intro not
      by_contra nothing
      apply not
      intro tr equal
      by_contra no
      exact nothing ⟨tr, equal, no⟩

end CompressionLean.Shrink
