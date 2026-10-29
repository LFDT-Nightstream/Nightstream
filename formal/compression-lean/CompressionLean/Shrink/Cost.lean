import CompressionLean.Shrink.Soundness

namespace CompressionLean.Shrink

open CompressionLean.Lib Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

def shrinkCount (m n L : ℕ) : ℕ := m * L + m * (L * 8) + L * 7 + (n + 1) * (L * 2)

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

end CompressionLean.Shrink
