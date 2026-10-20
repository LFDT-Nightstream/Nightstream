import CompressionLean.Shrink.Game

/-!
Owns what the shrink verifier's accepting paths give (`shrinkGame_avoids`):
along every path that avoids the bad sets, an accepted transcript has a
candidate that satisfies its program's CCS.

The chain runs backwards from the last check, for the candidate that the
opening names (`programGame_avoids`):

- the inner check and the opening make the inner rounds' last claim true,
  so their first claim, the batch `Σ_j ρ^j·v_j`, is true;
- off the roots of `batchPoly`, the claimed values are the true `M̃_j·z(rx)`;
- the outer check then makes the outer rounds' last claim true, so their
  first claim `0` is the zero-check sum, the extension at `τ` of `P` on the
  rows;
- off `foldBad` at each coordinate of `τ`, that sum is nonzero unless `P`
  vanishes on every row, so the CCS holds.
-/

namespace CompressionLean.Shrink

open CompressionLean.Layer1 Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

local notation "ι" => algebraMap F K

omit [Fintype K] [DecidableEq K] in
/-- The nodes `0, ..., d + 1` for `d ≤ 7` are distinct when the integers
below 9 are. -/
theorem shrinkNodes (cast : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b) (d : ℕ) (small : d ≤ 7) :
    NodesDistinct K d :=
  fun i j equal => Fin.ext (cast i j (by omega) (by omega) equal)

/-- Along every path that avoids the bad sets, an accepted transcript names
this program and has a candidate that satisfies `C`, when the shape check
puts `C` inside the cubes. -/
theorem programGame_avoids {I : Type} (i : I) (C : Ccs F) (shapeOk : Prop) (L : ℕ)
    (fits : shapeOk → C.Fits) (nodes7 : NodesDistinct K 7) (nodes1 : NodesDistinct K 1) :
    (programGame (K := K) i C shapeOk L).Avoids fun r =>
      ∀ tr, r = some tr → tr.inst = i ∧ ∃ w ∈ tr.cands, C.Relation w := by
  intro cands
  rw [Game.avoids_bind]
  refine (drawPoint_avoids (tauBad C cands.1) C.rowVars []).mono ?_
  rintro tau ⟨tauCs, tauEq, tauLength, tauPath⟩
  simp only [List.nil_append] at tauEq
  subst tauEq
  rw [Game.avoids_bind]
  refine (sumcheck_avoids Finset.univ 7 nodes7 _ _ (fun _ => ∅) C.rowVars
    (fun k _ => combRound_truth C.rowVars 7 (outerTables C.entries (C.z (cands.1.get k)) tau) outerComb
      outerComb_degree) C.rowVars [] 0 (by simp)).mono ?_
  rintro ⟨rx, outerLast⟩ ⟨⟨rxCs, rxEq, rxLength, -⟩, outerTransfer⟩ values
  dsimp only at rxEq outerTransfer ⊢
  split_ifs with outerCheck
  swap
  · intro tr none
    cases none
  intro rho rhoOut
  rw [Game.avoids_bind]
  refine (sumcheck_avoids Finset.univ 1 nodes1 _ _ (fun _ => ∅) (C.cellVars + 1)
    (fun k _ => combRound_truth (C.cellVars + 1) 1 (innerTables C.entries (C.z (cands.1.get k)) rx rho)
      productComb productComb_degree) (C.cellVars + 1) [] (batch rho values) (by simp)).mono ?_
  rintro ⟨ry, innerLast⟩ ⟨⟨ryCs, ryEq, ryLength, -⟩, innerTransfer⟩ opened
  dsimp only at ryEq innerTransfer ⊢
  split_ifs with accept
  swap
  · intro tr none
    cases none
  rintro tr ⟨⟩
  obtain ⟨shape, ⟨w, member, openEq⟩, innerCheck⟩ := accept
  refine ⟨rfl, w, member, ?_⟩
  obtain ⟨k, rfl⟩ := List.mem_iff_get.mp member
  have fit := fits shape
  have rxLen : rx.length = C.rowVars := by rw [rxEq]; simpa using rxLength
  have ryLen : ry.length = C.cellVars + 1 := by rw [ryEq]; simpa using ryLength
  have rowsRx : ∀ e ∈ C.entries, e.row < 2 ^ rx.length := by rw [rxLen]; exact fit.row_lt
  have colsRy : ∀ e ∈ C.entries, e.col < 2 ^ ry.length := by rw [ryLen]; exact fit.col_lt
  -- The inner check and the opening make the inner rounds end true, so they start true.
  have innerEnd : innerLast =
      combTruth (C.cellVars + 1) (innerTables C.entries (C.z (cands.1.get k)) rx rho) productComb ry := by
    rw [← ryLen, inner_truth_final _ _ _ _ _ colsRy, mle_z C _ ry ryLen fit.statement_le, innerCheck,
      zValue, openEq]
  have innerStart : batch rho values =
      combTruth (C.cellVars + 1) (innerTables C.entries (C.z (cands.1.get k)) rx rho) productComb [] := by
    by_contra wrong
    exact innerTransfer k (Finset.mem_univ k) wrong innerEnd
  rw [inner_truth_nil _ _ _ _ _ rowsRx fit.col_lt] at innerStart
  -- Off the batching roots, the claimed values are the true ones.
  have valuesTrue : values = trueValues C (cands.1.get k) rx := by
    by_contra differ
    refine batch_ne differ (fun hit => rhoOut ?_) innerStart
    exact Finset.mem_biUnion.mpr ⟨k, Finset.mem_univ k, hit⟩
  -- The outer check then makes the outer rounds end true, so they start true.
  have outerEnd : outerLast =
      combTruth C.rowVars (outerTables C.entries (C.z (cands.1.get k)) tau) outerComb rx := by
    rw [← rxLen, outer_truth_final _ _ _ _ (rxLen.trans tauLength.symm), outerCheck, valuesTrue]
    rfl
  have outerStart : (0 : K) =
      combTruth C.rowVars (outerTables C.entries (C.z (cands.1.get k)) tau) outerComb [] := by
    by_contra wrong
    exact outerTransfer k (Finset.mem_univ k) wrong outerEnd
  rw [outer_truth_nil _ _ _ _ tauLength] at outerStart
  -- Off `foldBad`, a row where `P` fails would make the zero-check sum nonzero.
  intro r inside
  by_contra failing
  have path : Path (foldBad (constraintTable C.entries (C.z (cands.1.get k))) tau.length) [] tau := by
    rw [tauLength]
    exact Path.mono (fun q _ hit => Finset.mem_biUnion.mpr ⟨k, Finset.mem_univ k, hit⟩) [] tau tauPath
  have nonzero : NonzeroOn tau.length (constraintTable C.entries (C.z (cands.1.get k))) :=
    ⟨r, by rw [tauLength]; exact inside,
      fun zero => failing ((map_eq_zero_iff (algebraMap F K) (algebraMap F K).injective).mp zero)⟩
  exact mle_ne_zero_of_path' _ tau nonzero path outerStart.symm

/-- **Shrink soundness, paths.** Along every path that avoids the bad sets,
an accepted transcript has a candidate that satisfies its program's CCS. -/
theorem shrinkGame_avoids {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F) (L : ℕ)
    (wf : T.WellFormed) (valid : ∀ i, (prog i).Valid T) (nodes7 : NodesDistinct K 7)
    (nodes1 : NodesDistinct K 1) :
    (shrinkGame (K := K) T s prog L).Avoids fun r =>
      ∀ tr, r = some tr → ∃ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w := by
  intro i
  refine (programGame_avoids i ((prog i).ccs T s) ((prog i).counts = s) L (ccs_fits T s (prog i) wf (valid i))
    nodes7 nodes1).mono ?_
  intro r all tr equal
  obtain ⟨same, holds⟩ := all tr equal
  rw [same]
  exact holds

end CompressionLean.Shrink
