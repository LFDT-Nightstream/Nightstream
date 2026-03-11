import SuperNeo.Ring

/-!
Embedding layer (P9 core):
- element/vector/matrix embedding and unembedding,
- blockwise add/scale operators,
- basic linearity theorems for `embedVec`.
-/

namespace SuperNeo

open F
local instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

/-- Element embedding: `F^d -> Coeffs`. In the compact scaffold this is identity. -/
def embedElem (v : Array F) : Coeffs :=
  v

/-- Element unembedding: `Coeffs -> F^d`. In the compact scaffold this is identity. -/
def unembedElem (a : Coeffs) : Array F :=
  a

@[simp] theorem unembedElem_embedElem (v : Array F) :
    unembedElem (embedElem v) = v := by
  rfl

@[simp] theorem embedElem_unembedElem (a : Coeffs) :
    embedElem (unembedElem a) = a := by
  rfl

theorem embedElem_vecAdd (v w : Array F) :
    embedElem (vecAdd v w) = vecAdd (embedElem v) (embedElem w) := by
  rfl

theorem unembedElem_vecAdd (a b : Coeffs) :
    unembedElem (vecAdd a b) = vecAdd (unembedElem a) (unembedElem b) := by
  rfl

theorem embedElem_vecScale (s : F) (v : Array F) :
    embedElem (vecScale s v) = vecScale s (embedElem v) := by
  rfl

theorem unembedElem_vecScale (s : F) (a : Coeffs) :
    unembedElem (vecScale s a) = vecScale s (unembedElem a) := by
  rfl

private def chunkExact (xs : Array F) (chunk : Nat) : Array (Array F) :=
  if chunk = 0 then
    #[]
  else
    Array.ofFn (fun t : Fin (xs.size / chunk) =>
      xs.extract (t.1 * chunk) (t.1 * chunk + chunk))

private def flatten (blocks : Array (Array F)) : Array F :=
  blocks.foldl (fun acc blk => acc ++ blk) #[]

/-- Blockwise addition on vectors of coefficient blocks. -/
def vecAddBlocks (a b : Array Coeffs) : Array Coeffs :=
  if hSize : a.size = b.size then
    Array.ofFn (fun i : Fin a.size =>
      vecAdd (a[i.1]'i.2) (b[i.1]'(by simpa [hSize] using i.2)))
  else
    #[]

/-- Blockwise scalar multiplication on vectors of coefficient blocks. -/
def vecScaleBlocks (s : F) (a : Array Coeffs) : Array Coeffs :=
  a.map (vecScale s)

theorem vecAddBlocks_size_of_eq
  {a b : Array Coeffs}
  (hSize : a.size = b.size) :
  (vecAddBlocks a b).size = a.size := by
  unfold vecAddBlocks
  simp [hSize]

theorem vecScaleBlocks_size
  (s : F) (a : Array Coeffs) :
  (vecScaleBlocks s a).size = a.size := by
  unfold vecScaleBlocks
  simp

/-- Vector embedding by `d`-chunking. -/
def embedVec (z : Array F) : Array Coeffs :=
  if z.size % d != 0 then
    #[]
  else
    (chunkExact z d).map embedElem

/-- Vector unembedding by block flattening. -/
def unembedVec (zr : Array Coeffs) : Array F :=
  Array.ofFn (fun i : Fin (zr.size * d) =>
    let hBlock : i.1 / d < zr.size := by
      exact (Nat.div_lt_iff_lt_mul d_pos).2 (by
        simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using i.2)
    (unembedElem (zr[i.1 / d]'hBlock)).getD (i.1 % d) 0)

@[simp] theorem unembedVec_size (zr : Array Coeffs) :
    (unembedVec zr).size = zr.size * d := by
  simp [unembedVec]

private theorem d_ne_zero : d ≠ 0 := by
  unfold d
  decide

private theorem chunkExact_size
  (xs : Array F) (chunk : Nat) (hChunk : chunk ≠ 0) :
  (chunkExact xs chunk).size = xs.size / chunk := by
  unfold chunkExact
  simp [hChunk]

private theorem vecScale_extract
  (s : F) (z : Array F) (start stop : Nat) :
  (vecScale s z).extract start stop = vecScale s (z.extract start stop) := by
  simp [vecScale]

private theorem vecAdd_eq_zipWith_of_size_eq
  {v w : Array F}
  (hSize : v.size = w.size) :
  vecAdd v w = Array.zipWith (fun x y => x + y) v w := by
  apply Array.ext
  · simp [vecAdd, hSize]
  · intro i hiL hiR
    have hiV : i < v.size := by
      simpa [hSize] using hiR
    have hiW : i < w.size := by
      simpa [hSize] using hiV
    simp [vecAdd, hSize]

private theorem vecAdd_extract_of_size_eq
  {v w : Array F}
  (hSize : v.size = w.size)
  (start stop : Nat) :
  (vecAdd v w).extract start stop =
    vecAdd (v.extract start stop) (w.extract start stop) := by
  calc
    (vecAdd v w).extract start stop
        = (Array.zipWith (fun x y => x + y) v w).extract start stop := by
            simp [vecAdd_eq_zipWith_of_size_eq hSize]
    _ = Array.zipWith (fun x y => x + y) (v.extract start stop) (w.extract start stop) := by
          simpa using
            (Array.extract_zipWith (f := fun x y => x + y) (as := v) (bs := w)
              (i := start) (j := stop))
    _ = vecAdd (v.extract start stop) (w.extract start stop) := by
          have hExtractSize : (v.extract start stop).size = (w.extract start stop).size := by
            simp [hSize]
          simp [vecAdd_eq_zipWith_of_size_eq hExtractSize]

theorem unembedVec_embedVec_of_mod_eq_zero
  {z : Array F}
  (hMod : z.size % d = 0) :
  unembedVec (embedVec z) = z := by
  have hNe : (z.size % d != 0) = false := by
    simp [hMod]
  have hDivMul : (z.size / d) * d = z.size := by
    exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hMod)
  unfold unembedVec embedVec
  simp [hNe, chunkExact, d_ne_zero]
  apply Array.ext
  · simpa [hMod, hDivMul, chunkExact, d_ne_zero]
  · intro i hiL hiR
    have hlt : i < (z.size / d) * d := by
      simpa [hDivMul] using hiR
    have hBlockLt : i / d < z.size / d := by
      exact (Nat.div_lt_iff_lt_mul d_pos).2 hlt
    have hBlockSuccLe : i / d + 1 ≤ z.size / d := Nat.succ_le_of_lt hBlockLt
    have hStopLe : (i / d) * d + d ≤ z.size := by
      have hMul : (i / d + 1) * d ≤ (z.size / d) * d :=
        Nat.mul_le_mul_right d hBlockSuccLe
      have hMul' : (i / d) * d + d ≤ (z.size / d) * d := by
        simpa [Nat.succ_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hMul
      simpa [hDivMul] using hMul'
    have hExtractSize :
        (z.extract ((i / d) * d) ((i / d) * d + d)).size = d := by
      simp [Array.size_extract, Nat.min_eq_left hStopLe, d_ne_zero]
    have hOffLt : i % d < (z.extract ((i / d) * d) ((i / d) * d + d)).size := by
      simpa [hExtractSize] using (Nat.mod_lt i d_pos)
    have hExtract :
        (z.extract ((i / d) * d) ((i / d) * d + d))[i % d] =
          z[(i / d) * d + (i % d)] :=
      Array.getElem_extract (xs := z) (start := (i / d) * d) (stop := (i / d) * d + d)
        (i := i % d) hOffLt
    have hExtractSome :
        (z.extract ((i / d) * d) ((i / d) * d + d))[i % d]? =
          some z[(i / d) * d + (i % d)] := by
      simpa [Array.getElem?_eq_getElem hOffLt] using congrArg some hExtract
    have hLeft :
        (z.extract ((i / d) * d) ((i / d) * d + d))[i % d]?.getD 0 =
          z[(i / d) * d + (i % d)] := by
      simpa [hExtractSome]
    have hIdx : (i / d) * d + (i % d) = i := by
      simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using (Nat.div_add_mod i d)
    simpa [hIdx] using hLeft

theorem unembedVec_vecScaleBlocks
  (s : F)
  (zr : Array Coeffs) :
  unembedVec (vecScaleBlocks s zr) = vecScale s (unembedVec zr) := by
  apply Array.ext
  · simp [unembedVec, vecScaleBlocks, vecScale]
  · intro i hiL hiR
    have hBlock : i / d < zr.size := by
      exact (Nat.div_lt_iff_lt_mul d_pos).2 (by
        simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hiR)
    have hBlockScaled : i / d < (vecScaleBlocks s zr).size := by
      simpa [vecScaleBlocks] using hBlock
    have hPoint :
        (unembedElem (Array.map (fun x => s * x) zr[i / d]))[i % d]?.getD 0 =
          s * (unembedElem zr[i / d])[i % d]?.getD 0 := by
      have hMapGetD :
          (Option.map (fun x => s * x) zr[i / d][i % d]?).getD 0 =
            s * zr[i / d][i % d]?.getD 0 := by
        cases hOpt : zr[i / d][i % d]? with
        | none =>
            simpa using (Fin.mul_zero (n := Goldilocks.q) s).symm
        | some x =>
            simp [hOpt]
      simpa [unembedElem] using hMapGetD
    simpa [unembedVec, vecScaleBlocks, vecScale, hBlock, hBlockScaled] using hPoint

theorem unembedVec_vecAddBlocks_of_size_eq_of_block_size_d
  {a b : Array Coeffs}
  (hSize : a.size = b.size)
  (ha : ∀ i : Fin a.size, (a[i.1]'i.2).size = d)
  (hb : ∀ i : Fin b.size, (b[i.1]'i.2).size = d) :
  unembedVec (vecAddBlocks a b) = vecAdd (unembedVec a) (unembedVec b) := by
  have hUnembedSizeEq : (unembedVec a).size = (unembedVec b).size := by
    simp [unembedVec, hSize]
  have hLSize : (unembedVec (vecAddBlocks a b)).size = a.size * d := by
    calc
      (unembedVec (vecAddBlocks a b)).size
          = (vecAddBlocks a b).size * d := by simp [unembedVec]
      _ = a.size * d := by simpa [vecAddBlocks_size_of_eq hSize]
  have hRSize : (vecAdd (unembedVec a) (unembedVec b)).size = a.size * d := by
    calc
      (vecAdd (unembedVec a) (unembedVec b)).size
          = (unembedVec a).size := vecAdd_size_of_eq hUnembedSizeEq
      _ = a.size * d := by simp [unembedVec]
  apply Array.ext
  · exact hLSize.trans hRSize.symm
  · intro i hiL hiR
    have hi : i < a.size * d := by simpa [hRSize] using hiR
    have hBlockA : i / d < a.size := by
      exact (Nat.div_lt_iff_lt_mul d_pos).2 (by simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hi)
    have hBlockB : i / d < b.size := by
      simpa [hSize] using hBlockA
    have hBlockAB : i / d < (vecAddBlocks a b).size := by
      simpa [vecAddBlocks_size_of_eq hSize] using hBlockA
    have hOffLt : i % d < d := Nat.mod_lt i d_pos
    have hAblock : (a[i / d]'hBlockA).size = d := ha ⟨i / d, hBlockA⟩
    have hBblock : (b[i / d]'hBlockB).size = d := hb ⟨i / d, hBlockB⟩
    have hCoeff :
        coeffAt (vecAdd (a[i / d]'hBlockA) (b[i / d]'hBlockB)) (i % d) =
          coeffAt (a[i / d]'hBlockA) (i % d) + coeffAt (b[i / d]'hBlockB) (i % d) := by
      exact coeffAt_vecAdd_of_size_d
        (a[i / d]'hBlockA) (b[i / d]'hBlockB) hAblock hBblock (i % d) hOffLt
    have hGet :
        (vecAdd (a[i / d]'hBlockA) (b[i / d]'hBlockB)).getD (i % d) 0 =
          (a[i / d]'hBlockA).getD (i % d) 0 + (b[i / d]'hBlockB).getD (i % d) 0 := by
      simpa [coeffAt, hOffLt] using hCoeff
    have hLeft :
        (unembedVec (vecAddBlocks a b))[i] =
          (a[i / d]'hBlockA).getD (i % d) 0 + (b[i / d]'hBlockB).getD (i % d) 0 := by
      -- unfold through block-unembedding and vecAddBlocks.
      simpa [unembedVec, vecAddBlocks, hSize, hBlockAB, unembedElem] using hGet
    have hRight :
        (vecAdd (unembedVec a) (unembedVec b))[i] =
          (a[i / d]'hBlockA).getD (i % d) 0 + (b[i / d]'hBlockB).getD (i % d) 0 := by
      have hSizeMul : a.size * d = b.size * d := by
        simpa [hSize]
      unfold vecAdd
      simp [hSizeMul, unembedVec, hBlockA, hBlockB, unembedElem]
    exact hLeft.trans hRight.symm

private theorem chunkExact_vecScale
  (s : F) (z : Array F) :
  chunkExact (vecScale s z) d = (chunkExact z d).map (vecScale s) := by
  unfold chunkExact
  simp [d_ne_zero]
  apply Array.ext
  · simp [vecScale_size]
  · intro i hiL hiR
    simp [vecScale_extract]

private theorem chunkExact_vecAdd
  {v w : Array F}
  (hSize : v.size = w.size) :
  chunkExact (vecAdd v w) d = vecAddBlocks (chunkExact v d) (chunkExact w d) := by
  have hVecAddSize : (vecAdd v w).size = v.size := vecAdd_size_of_eq hSize
  unfold chunkExact vecAddBlocks
  simp [d_ne_zero, hSize, hVecAddSize]
  apply Array.ext
  · simpa [hVecAddSize]
  · intro i hiL hiR
    have hExtract :
        (vecAdd v w).extract (i * d) (i * d + d) =
          vecAdd (v.extract (i * d) (i * d + d)) (w.extract (i * d) (i * d + d)) :=
      vecAdd_extract_of_size_eq hSize (i * d) (i * d + d)
    simp [hExtract]

private theorem map_embedElem_vecAddBlocks
  (a b : Array Coeffs) :
  Array.map embedElem (vecAddBlocks a b) =
    vecAddBlocks (Array.map embedElem a) (Array.map embedElem b) := by
  unfold vecAddBlocks
  by_cases hSize : a.size = b.size
  · simp [hSize]
    apply Array.ext
    · simp
    · intro i hiL hiR
      simp [embedElem]
  · simp [hSize, embedElem]

theorem embedVec_vecScale_of_mod_eq_zero
  {z : Array F}
  (hMod : z.size % d = 0)
  (s : F) :
  embedVec (vecScale s z) = vecScaleBlocks s (embedVec z) := by
  have hScaleMod : (vecScale s z).size % d = 0 := by
    simpa [vecScale_size] using hMod
  unfold embedVec vecScaleBlocks
  simp [hMod, hScaleMod, embedElem, chunkExact_vecScale]

theorem embedVec_vecAdd_of_size_mod_eq_zero
  {v w : Array F}
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  embedVec (vecAdd v w) = vecAddBlocks (embedVec v) (embedVec w) := by
  have hModW : w.size % d = 0 := by
    simpa [hSize] using hMod
  have hAddMod : (vecAdd v w).size % d = 0 := by
    simpa [vecAdd_size_of_eq hSize] using hMod
  unfold embedVec
  simp [hAddMod, hMod, hModW, chunkExact_vecAdd hSize, map_embedElem_vecAddBlocks]

/-- Every embedded block has canonical ring length `d` when `z.size % d = 0`. -/
theorem embedVec_block_size_of_mod_eq_zero
  {z : Array F}
  (hMod : z.size % d = 0)
  (i : Fin (embedVec z).size) :
  ((embedVec z)[i.1]'i.2).size = d := by
  have hNe : (z.size % d != 0) = false := by
    simp [hMod]
  have hDivMul : (z.size / d) * d = z.size := by
    exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hMod)
  have hi : i.1 < z.size / d := by
    simpa [embedVec, hNe, chunkExact, d_ne_zero] using i.2
  have hStopLe : i.1 * d + d ≤ z.size := by
    have hSucc : i.1 + 1 ≤ z.size / d := Nat.succ_le_of_lt hi
    have hMul : (i.1 + 1) * d ≤ (z.size / d) * d := Nat.mul_le_mul_right d hSucc
    have hMul' : i.1 * d + d ≤ (z.size / d) * d := by
      simpa [Nat.succ_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hMul
    simpa [hDivMul] using hMul'
  have hExtractSize :
      (z.extract (i.1 * d) (i.1 * d + d)).size = d := by
    simp [Array.size_extract, Nat.min_eq_left hStopLe, d_ne_zero]
  simpa [embedVec, hNe, chunkExact, d_ne_zero, embedElem] using hExtractSize

/-- Matrix embedding row-wise. -/
def embedMatrix (m : Array (Array F)) : Array (Array Coeffs) :=
  m.map embedVec

/-- Matrix unembedding row-wise. -/
def unembedMatrix (mr : Array (Array Coeffs)) : Array (Array F) :=
  mr.map unembedVec

/-- Row-wise matrix scaling on field vectors. -/
def matrixScaleRows (s : F) (m : Array (Array F)) : Array (Array F) :=
  m.map (vecScale s)

/-- Row-wise matrix scaling on embedded vectors. -/
def matrixScaleRowsBlocks (s : F) (mr : Array (Array Coeffs)) : Array (Array Coeffs) :=
  mr.map (vecScaleBlocks s)

/-- Row-wise matrix addition on field vectors. -/
def matrixAddRows (m n : Array (Array F)) : Array (Array F) :=
  if h : m.size = n.size then
    Array.ofFn (fun i : Fin m.size =>
      vecAdd (m[i.1]'i.2) (n[i.1]'(by simpa [h] using i.2)))
  else
    #[]

/-- Row-wise matrix addition on embedded vectors. -/
def matrixAddRowsBlocks (m n : Array (Array Coeffs)) : Array (Array Coeffs) :=
  if h : m.size = n.size then
    Array.ofFn (fun i : Fin m.size =>
      vecAddBlocks (m[i.1]'i.2) (n[i.1]'(by simpa [h] using i.2)))
  else
    #[]

theorem embedMatrix_rowwise_vecScale_of_rows_mod_eq_zero
  {m : Array (Array F)}
  (hRowsMod : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0)
  (s : F) :
  embedMatrix (matrixScaleRows s m) =
    matrixScaleRowsBlocks s (embedMatrix m) := by
  unfold embedMatrix matrixScaleRows matrixScaleRowsBlocks
  apply Array.ext
  · simp
  · intro i hiL hiR
    have hi : i < m.size := by
      simpa using hiL
    have hRowMod : (m[i]'hi).size % d = 0 := hRowsMod ⟨i, hi⟩
    simpa [hi] using
      (embedVec_vecScale_of_mod_eq_zero (z := m[i]'hi) hRowMod s)

theorem embedMatrix_rowwise_vecAdd_of_rows_size_mod_eq_zero
  {m n : Array (Array F)}
  (hRowsSize : m.size = n.size)
  (hRowEq :
    ∀ i : Fin m.size, (m[i.1]'i.2).size = (n[i.1]'(by simpa [hRowsSize] using i.2)).size)
  (hRowsMod : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0) :
  embedMatrix (matrixAddRows m n) =
    matrixAddRowsBlocks (embedMatrix m) (embedMatrix n) := by
  unfold embedMatrix matrixAddRows matrixAddRowsBlocks
  simp [hRowsSize]
  apply Array.ext
  · simp
  · intro i hiL hiR
    have hi : i < m.size := by
      simpa using hiL
    have hMod : (m[i]'hi).size % d = 0 := hRowsMod ⟨i, hi⟩
    have hSize : (m[i]'hi).size = (n[i]'(by simpa [hRowsSize] using hi)).size := by
      exact hRowEq ⟨i, hi⟩
    simpa [hi] using
      (embedVec_vecAdd_of_size_mod_eq_zero
        (v := m[i]'hi)
        (w := n[i]'(by simpa [hRowsSize] using hi))
        hSize hMod)

theorem unembedMatrix_embedMatrix_of_rows_mod_eq_zero
  {m : Array (Array F)}
  (hRowsMod : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0) :
  unembedMatrix (embedMatrix m) = m := by
  unfold unembedMatrix embedMatrix
  apply Array.ext
  · simp
  · intro i hiL hiR
    have hi : i < m.size := by
      simpa using hiR
    have hMod : (m[i]'hi).size % d = 0 := hRowsMod ⟨i, hi⟩
    simpa [hi] using
      (unembedVec_embedVec_of_mod_eq_zero (z := m[i]'hi) hMod)

/-! ## Round-trip executable checks + proposition bridges -/

/-- Bool-level vector embedding round-trip check. -/
def embeddingVecRoundTrip (z : Array F) : Bool :=
  if z.size % d != 0 then
    false
  else
    decide (unembedVec (embedVec z) = z)

/-- Proposition-level counterpart of `embeddingVecRoundTrip`. -/
def embeddingVecRoundTripProp (z : Array F) : Prop :=
  (z.size % d != 0) = false ∧
    unembedVec (embedVec z) = z

theorem embeddingVecRoundTrip_sound
  {z : Array F}
  (hOk : embeddingVecRoundTrip z = true) :
  embeddingVecRoundTripProp z := by
  unfold embeddingVecRoundTrip at hOk
  cases hSize : (z.size % d != 0) with
  | true =>
      simp [hSize] at hOk
  | false =>
      simp [hSize] at hOk
      exact ⟨hSize, hOk⟩

theorem embeddingVecRoundTrip_complete
  {z : Array F}
  (hProp : embeddingVecRoundTripProp z) :
  embeddingVecRoundTrip z = true := by
  rcases hProp with ⟨hSize, hEq⟩
  unfold embeddingVecRoundTrip
  simp [hSize, decide_eq_true hEq]

theorem embeddingVecRoundTrip_iff_prop
  {z : Array F} :
  embeddingVecRoundTrip z = true ↔ embeddingVecRoundTripProp z := by
  constructor
  · exact embeddingVecRoundTrip_sound
  · exact embeddingVecRoundTrip_complete

theorem embeddingVecRoundTrip_size_mod_eq_zero
  {z : Array F}
  (hOk : embeddingVecRoundTrip z = true) :
  z.size % d = 0 := by
  have hSizeFalse : (z.size % d != 0) = false := (embeddingVecRoundTrip_sound hOk).1
  by_cases hMod : z.size % d = 0
  · exact hMod
  · have hNeTrue : (z.size % d != 0) = true := by simp [hMod]
    rw [hNeTrue] at hSizeFalse
    cases hSizeFalse

theorem embeddingVecRoundTrip_unembed_embed_eq
  {z : Array F}
  (hOk : embeddingVecRoundTrip z = true) :
  unembedVec (embedVec z) = z := by
  exact (embeddingVecRoundTrip_sound hOk).2

/-- Bool-level matrix embedding round-trip check. -/
def embeddingMatrixRoundTrip (m : Array (Array F)) : Bool :=
  if !(m.all (fun row => row.size % d = 0)) then
    false
  else
    decide (unembedMatrix (embedMatrix m) = m)

/-- Proposition-level counterpart of `embeddingMatrixRoundTrip`. -/
def embeddingMatrixRoundTripProp (m : Array (Array F)) : Prop :=
  m.all (fun row => row.size % d = 0) = true ∧
    unembedMatrix (embedMatrix m) = m

theorem embeddingMatrixRoundTrip_sound
  {m : Array (Array F)}
  (hOk : embeddingMatrixRoundTrip m = true) :
  embeddingMatrixRoundTripProp m := by
  unfold embeddingMatrixRoundTrip at hOk
  cases hAll : m.all (fun row => row.size % d = 0) with
  | false =>
      simp [hAll] at hOk
  | true =>
      simp [hAll] at hOk
      exact ⟨hAll, hOk⟩

theorem embeddingMatrixRoundTrip_complete
  {m : Array (Array F)}
  (hProp : embeddingMatrixRoundTripProp m) :
  embeddingMatrixRoundTrip m = true := by
  rcases hProp with ⟨hAll, hEq⟩
  unfold embeddingMatrixRoundTrip
  simp [hAll, decide_eq_true hEq]

theorem embeddingMatrixRoundTrip_iff_prop
  {m : Array (Array F)} :
  embeddingMatrixRoundTrip m = true ↔ embeddingMatrixRoundTripProp m := by
  constructor
  · exact embeddingMatrixRoundTrip_sound
  · exact embeddingMatrixRoundTrip_complete

theorem embeddingMatrixRoundTrip_rows_mod_ok
  {m : Array (Array F)}
  (hOk : embeddingMatrixRoundTrip m = true) :
  m.all (fun row => row.size % d = 0) = true := by
  exact (embeddingMatrixRoundTrip_sound hOk).1

theorem embeddingMatrixRoundTrip_unembed_embed_eq
  {m : Array (Array F)}
  (hOk : embeddingMatrixRoundTrip m = true) :
  unembedMatrix (embedMatrix m) = m := by
  exact (embeddingMatrixRoundTrip_sound hOk).2

/-! ## Theorem-facing P9 package surfaces -/

/-- Theorem-facing element embedding package (Definition 7 interface). -/
def p9ElemEmbeddingAssumption : Prop :=
  (∀ v : Array F, unembedElem (embedElem v) = v) ∧
  (∀ a : Coeffs, embedElem (unembedElem a) = a) ∧
  (∀ v w : Array F, embedElem (vecAdd v w) = vecAdd (embedElem v) (embedElem w)) ∧
  (∀ s : F, ∀ v : Array F, embedElem (vecScale s v) = vecScale s (embedElem v))

theorem p9ElemEmbeddingAssumption_from_defs :
  p9ElemEmbeddingAssumption := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro v
    exact unembedElem_embedElem v
  · intro a
    exact embedElem_unembedElem a
  · intro v w
    exact embedElem_vecAdd v w
  · intro s v
    exact embedElem_vecScale s v

/-- Theorem-facing vector embedding package (Definition 7 interface). -/
def p9VecEmbeddingAssumption : Prop :=
  ∀ z : Array F, z.size % d = 0 → unembedVec (embedVec z) = z

/-- Check-facing vector embedding package (regression compatibility). -/
def p9VecEmbeddingCheckAssumption : Prop :=
  ∀ z : Array F, z.size % d = 0 → embeddingVecRoundTrip z = true

/-- Theorem-facing matrix embedding package (row-wise Definition 7 interface). -/
def p9MatrixEmbeddingAssumption : Prop :=
  ∀ m : Array (Array F),
    m.all (fun row => row.size % d = 0) = true →
      unembedMatrix (embedMatrix m) = m

/-- Check-facing matrix embedding package (regression compatibility). -/
def p9MatrixEmbeddingCheckAssumption : Prop :=
  ∀ m : Array (Array F),
    m.all (fun row => row.size % d = 0) = true →
      embeddingMatrixRoundTrip m = true

theorem p9VecEmbeddingAssumption_of_checkAssumption
  (hCheck : p9VecEmbeddingCheckAssumption) :
  p9VecEmbeddingAssumption := by
  intro z hMod
  exact (embeddingVecRoundTrip_sound (hCheck z hMod)).2

theorem p9VecEmbeddingCheckAssumption_of_assumption
  (hAssm : p9VecEmbeddingAssumption) :
  p9VecEmbeddingCheckAssumption := by
  intro z hMod
  exact embeddingVecRoundTrip_complete ⟨by simp [hMod], hAssm z hMod⟩

theorem p9VecEmbeddingAssumption_iff_checkAssumption :
  p9VecEmbeddingAssumption ↔ p9VecEmbeddingCheckAssumption := by
  constructor
  · exact p9VecEmbeddingCheckAssumption_of_assumption
  · exact p9VecEmbeddingAssumption_of_checkAssumption

theorem p9VecEmbeddingAssumption_holds :
  p9VecEmbeddingAssumption := by
  intro z hMod
  exact unembedVec_embedVec_of_mod_eq_zero hMod

theorem p9MatrixEmbeddingAssumption_of_checkAssumption
  (hCheck : p9MatrixEmbeddingCheckAssumption) :
  p9MatrixEmbeddingAssumption := by
  intro m hRows
  exact (embeddingMatrixRoundTrip_sound (hCheck m hRows)).2

theorem p9MatrixEmbeddingCheckAssumption_of_assumption
  (hAssm : p9MatrixEmbeddingAssumption) :
  p9MatrixEmbeddingCheckAssumption := by
  intro m hRows
  exact embeddingMatrixRoundTrip_complete ⟨hRows, hAssm m hRows⟩

theorem p9MatrixEmbeddingAssumption_iff_checkAssumption :
  p9MatrixEmbeddingAssumption ↔ p9MatrixEmbeddingCheckAssumption := by
  constructor
  · exact p9MatrixEmbeddingCheckAssumption_of_assumption
  · exact p9MatrixEmbeddingAssumption_of_checkAssumption

theorem p9MatrixEmbeddingAssumption_holds :
  p9MatrixEmbeddingAssumption := by
  intro m hRowsBool
  have hRows : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0 := by
    intro i
    have hDec : decide ((m[i.1]'i.2).size % d = 0) = true :=
      (Array.all_eq_true.mp hRowsBool) i.1 i.2
    exact decide_eq_true_eq.mp hDec
  exact unembedMatrix_embedMatrix_of_rows_mod_eq_zero hRows

/-- Combined theorem-facing P9 surface (element + vector + matrix). -/
def p9EmbeddingAssumption : Prop :=
  p9ElemEmbeddingAssumption ∧
    p9VecEmbeddingAssumption ∧
      p9MatrixEmbeddingAssumption

/-- Combined check-facing P9 surface (vector + matrix checks). -/
def p9EmbeddingCheckAssumption : Prop :=
  p9VecEmbeddingCheckAssumption ∧
    p9MatrixEmbeddingCheckAssumption

theorem p9EmbeddingAssumption_of_checkAssumption
  (hCheck : p9EmbeddingCheckAssumption) :
  p9EmbeddingAssumption := by
  exact ⟨
    p9ElemEmbeddingAssumption_from_defs,
    p9VecEmbeddingAssumption_of_checkAssumption hCheck.1,
    p9MatrixEmbeddingAssumption_of_checkAssumption hCheck.2
  ⟩

theorem p9EmbeddingCheckAssumption_of_assumption
  (hAssm : p9EmbeddingAssumption) :
  p9EmbeddingCheckAssumption := by
  exact ⟨
    p9VecEmbeddingCheckAssumption_of_assumption hAssm.2.1,
    p9MatrixEmbeddingCheckAssumption_of_assumption hAssm.2.2
  ⟩

theorem p9EmbeddingAssumption_iff_checkAssumption :
  p9EmbeddingCheckAssumption ↔
    (p9VecEmbeddingAssumption ∧ p9MatrixEmbeddingAssumption) := by
  constructor
  · intro hCheck
    exact ⟨
      p9VecEmbeddingAssumption_of_checkAssumption hCheck.1,
      p9MatrixEmbeddingAssumption_of_checkAssumption hCheck.2
    ⟩
  · intro hAssm
    exact ⟨
      p9VecEmbeddingCheckAssumption_of_assumption hAssm.1,
      p9MatrixEmbeddingCheckAssumption_of_assumption hAssm.2
    ⟩

theorem p9EmbeddingAssumption_elem
  (hAssm : p9EmbeddingAssumption) :
  p9ElemEmbeddingAssumption := by
  exact hAssm.1

theorem p9EmbeddingAssumption_vec
  (hAssm : p9EmbeddingAssumption) :
  p9VecEmbeddingAssumption := by
  exact hAssm.2.1

theorem p9EmbeddingAssumption_matrix
  (hAssm : p9EmbeddingAssumption) :
  p9MatrixEmbeddingAssumption := by
  exact hAssm.2.2

theorem p9EmbeddingAssumption_holds :
  p9EmbeddingAssumption := by
  exact ⟨
    p9ElemEmbeddingAssumption_from_defs,
    p9VecEmbeddingAssumption_holds,
    p9MatrixEmbeddingAssumption_holds
  ⟩

theorem embeddingVecRoundTrip_true_of_p9VecAssumption
  {z : Array F}
  (hAssm : p9VecEmbeddingAssumption)
  (hMod : z.size % d = 0) :
  embeddingVecRoundTrip z = true := by
  exact embeddingVecRoundTrip_complete ⟨by simp [hMod], hAssm z hMod⟩

theorem embeddingVecRoundTrip_true_of_mod
  {z : Array F}
  (hAssm : p9VecEmbeddingAssumption)
  (hMod : z.size % d = 0) :
  embeddingVecRoundTrip z = true := by
  exact embeddingVecRoundTrip_true_of_p9VecAssumption (hAssm := hAssm) hMod

theorem embeddingMatrixRoundTrip_true_of_p9MatrixAssumption
  {m : Array (Array F)}
  (hAssm : p9MatrixEmbeddingAssumption)
  (hRows : m.all (fun row => row.size % d = 0) = true) :
  embeddingMatrixRoundTrip m = true := by
  exact embeddingMatrixRoundTrip_complete ⟨hRows, hAssm m hRows⟩

theorem embeddingMatrixRoundTrip_true_of_rows_mod
  {m : Array (Array F)}
  (hAssm : p9MatrixEmbeddingAssumption)
  (hRows : m.all (fun row => row.size % d = 0) = true) :
  embeddingMatrixRoundTrip m = true := by
  exact embeddingMatrixRoundTrip_true_of_p9MatrixAssumption (hAssm := hAssm) hRows

theorem embeddingVecRoundTrip_true_of_p9EmbeddingAssumption
  {z : Array F}
  (hAssm : p9EmbeddingAssumption)
  (hMod : z.size % d = 0) :
  embeddingVecRoundTrip z = true := by
  exact embeddingVecRoundTrip_true_of_p9VecAssumption
    (hAssm := p9EmbeddingAssumption_vec hAssm) hMod

theorem embeddingMatrixRoundTrip_true_of_p9EmbeddingAssumption
  {m : Array (Array F)}
  (hAssm : p9EmbeddingAssumption)
  (hRows : m.all (fun row => row.size % d = 0) = true) :
  embeddingMatrixRoundTrip m = true := by
  exact embeddingMatrixRoundTrip_true_of_p9MatrixAssumption
    (hAssm := p9EmbeddingAssumption_matrix hAssm) hRows

/-- Small executable sanity check for embedding round-trip. -/
def embeddingSanity : Bool :=
  let z := ((List.range (2 * d)).toArray).map F.ofNat
  embeddingVecRoundTrip z

end SuperNeo
