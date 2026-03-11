import SuperNeo.Embedding

/-!
Contract interface for `SuperNeo.Embedding`.

Paper anchors:
- `./formal/superneo-lean/SuperNeo.pdf.md`, Definition 7 (Coefficient Embedding), lines 358-366
- `./formal/superneo-lean/SuperNeo.pdf.md`, Theorem 3 (Inner Product Transform), lines 368-372
- `./formal/superneo-lean/SuperNeo.pdf.md`, Theorem 5 (Evaluation Homomorphism), lines 390-400
-/

namespace SuperNeo

namespace EmbeddingInterface

/-! ## Element Embedding -/

/-- [Role: Theorem-Target] Element embedding: `F^d → Coeffs` (identity on coefficient type). -/
abbrev embedElem := SuperNeo.embedElem

/-- [Role: Theorem-Target] Element unembedding: `Coeffs → F^d` (identity on coefficient type). -/
abbrev unembedElem := SuperNeo.unembedElem

/-- [Role: Theorem-Target] `unembedElem(embedElem(v)) = v`. -/
abbrev unembedElem_embedElem := SuperNeo.unembedElem_embedElem

/-- [Role: Theorem-Target] `embedElem(unembedElem(a)) = a`. -/
abbrev embedElem_unembedElem := SuperNeo.embedElem_unembedElem

/-- [Role: Theorem-Target] `embedElem(v + w) = embedElem(v) + embedElem(w)`. -/
abbrev embedElem_vecAdd := SuperNeo.embedElem_vecAdd

/-- [Role: Theorem-Target] `embedElem(s·v) = s·embedElem(v)`. -/
abbrev embedElem_vecScale := SuperNeo.embedElem_vecScale

/-! ## Block Algebra -/

/-- [Role: Theorem-Target] Blockwise addition on coefficient-block vectors. -/
abbrev vecAddBlocks := SuperNeo.vecAddBlocks

/-- [Role: Theorem-Target] Blockwise scalar multiplication on coefficient-block vectors. -/
abbrev vecScaleBlocks := SuperNeo.vecScaleBlocks

/-- [Role: Theorem-Target] `(vecAddBlocks a b).size = a.size` when `a.size = b.size`. -/
theorem vecAddBlocks_size_of_eq
  {a b : Array Coeffs}
  (hSize : a.size = b.size) :
  (vecAddBlocks a b).size = a.size :=
  SuperNeo.vecAddBlocks_size_of_eq hSize

/-- [Role: Theorem-Target] `(vecScaleBlocks s a).size = a.size`. -/
abbrev vecScaleBlocks_size := SuperNeo.vecScaleBlocks_size

/-! ## Vector Embedding & Linearity -/

/-- [Role: Theorem-Target] Vector embedding: partition `F^{d·n_R}` into d-chunks, embed each as ring element. -/
abbrev embedVec := SuperNeo.embedVec

/-- [Role: Theorem-Target] Vector unembedding: flatten coefficient blocks back to field vector. -/
abbrev unembedVec := SuperNeo.unembedVec

/-- [Role: Theorem-Target] Scale linearity: `embedVec(s·z) = vecScaleBlocks s (embedVec z)` when `z.size % d = 0`. -/
theorem embedVec_vecScale_of_mod_eq_zero
  {z : Array F}
  (hMod : z.size % d = 0)
  (s : F) :
  embedVec (vecScale s z) = vecScaleBlocks s (embedVec z) :=
  SuperNeo.embedVec_vecScale_of_mod_eq_zero hMod s

/-- [Role: Theorem-Target] Add linearity: `embedVec(v+w) = vecAddBlocks (embedVec v) (embedVec w)` when sizes match and `v.size % d = 0`. -/
theorem embedVec_vecAdd_of_size_mod_eq_zero
  {v w : Array F}
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  embedVec (vecAdd v w) = vecAddBlocks (embedVec v) (embedVec w) :=
  SuperNeo.embedVec_vecAdd_of_size_mod_eq_zero hSize hMod

/-- [Role: Theorem-Target] Round-trip: `unembedVec(embedVec z) = z` when `z.size % d = 0`. -/
theorem unembedVec_embedVec_of_mod_eq_zero
  {z : Array F}
  (hMod : z.size % d = 0) :
  unembedVec (embedVec z) = z :=
  SuperNeo.unembedVec_embedVec_of_mod_eq_zero hMod

/-! ## Matrix Embedding & Linearity -/

/-- [Role: Theorem-Target] Matrix embedding: row-wise application of `embedVec`. -/
abbrev embedMatrix := SuperNeo.embedMatrix

/-- [Role: Theorem-Target] Matrix unembedding: row-wise application of `unembedVec`. -/
abbrev unembedMatrix := SuperNeo.unembedMatrix

/-- [Role: Theorem-Target] Matrix scale linearity (row-wise) when all rows satisfy `size % d = 0`. -/
theorem embedMatrix_rowwise_vecScale_of_rows_mod_eq_zero
  {m : Array (Array F)}
  (hRowsMod : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0)
  (s : F) :
  embedMatrix (matrixScaleRows s m) =
    matrixScaleRowsBlocks s (embedMatrix m) :=
  SuperNeo.embedMatrix_rowwise_vecScale_of_rows_mod_eq_zero hRowsMod s

/-- [Role: Theorem-Target] Matrix add linearity (row-wise) when row counts match, per-row sizes match, and `size % d = 0`. -/
theorem embedMatrix_rowwise_vecAdd_of_rows_size_mod_eq_zero
  {m n : Array (Array F)}
  (hRowsSize : m.size = n.size)
  (hRowEq :
    ∀ i : Fin m.size, (m[i.1]'i.2).size = (n[i.1]'(by simpa [hRowsSize] using i.2)).size)
  (hRowsMod : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0) :
  embedMatrix (matrixAddRows m n) =
    matrixAddRowsBlocks (embedMatrix m) (embedMatrix n) :=
  SuperNeo.embedMatrix_rowwise_vecAdd_of_rows_size_mod_eq_zero hRowsSize hRowEq hRowsMod

/-- [Role: Theorem-Target] Matrix round-trip: `unembedMatrix(embedMatrix m) = m` when all rows satisfy `size % d = 0`. -/
theorem unembedMatrix_embedMatrix_of_rows_mod_eq_zero
  {m : Array (Array F)}
  (hRowsMod : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0) :
  unembedMatrix (embedMatrix m) = m :=
  SuperNeo.unembedMatrix_embedMatrix_of_rows_mod_eq_zero hRowsMod

/-! ## Round-Trip Check/Prop Bridges -/

/-- [Role: Theorem-Target] `embeddingVecRoundTrip z = true ↔ embeddingVecRoundTripProp z`. -/
theorem embeddingVecRoundTrip_iff_prop
  {z : Array F} :
  embeddingVecRoundTrip z = true ↔ embeddingVecRoundTripProp z :=
  SuperNeo.embeddingVecRoundTrip_iff_prop

/-- [Role: Theorem-Target] `embeddingMatrixRoundTrip m = true ↔ embeddingMatrixRoundTripProp m`. -/
theorem embeddingMatrixRoundTrip_iff_prop
  {m : Array (Array F)} :
  embeddingMatrixRoundTrip m = true ↔ embeddingMatrixRoundTripProp m :=
  SuperNeo.embeddingMatrixRoundTrip_iff_prop

/-! ## Combined Package Closure -/

/-- [Role: Theorem-Target] Combined embedding package: element bijectivity + linearity, vector round-trip + linearity, matrix round-trip + linearity. -/
abbrev p9EmbeddingAssumption := SuperNeo.p9EmbeddingAssumption

/-- [Role: Theorem-Target] Discharges the combined embedding package from proved component theorems. -/
abbrev p9EmbeddingAssumption_holds := SuperNeo.p9EmbeddingAssumption_holds

end EmbeddingInterface

end SuperNeo
