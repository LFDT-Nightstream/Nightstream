import SuperNeo.Ring

/-!
Embedding layer (P9 core):
- element/vector/matrix embedding and unembedding,
- blockwise add/scale operators,
- basic linearity theorems for `embedVec`.
-/

namespace SuperNeo

open F

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
  flatten (zr.map unembedElem)

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

theorem embedVec_vecScale_of_mod_eq_zero
  {z : Array F}
  (hMod : z.size % d = 0)
  (s : F) :
  embedVec (vecScale s z) = vecScaleBlocks s (embedVec z) := by
  have hModScaled : (vecScale s z).size % d = 0 := by
    simpa [vecScale_size] using hMod
  unfold embedVec vecScaleBlocks
  simp [hMod]
  apply Array.ext
  · simp [chunkExact_size, d_ne_zero, vecScale_size]
  · intro i hiL hiR
    simp [chunkExact, d_ne_zero, vecScale_extract]
    rfl

theorem embedVec_vecAdd_of_size_mod_eq_zero
  {v w : Array F}
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  embedVec (vecAdd v w) = vecAddBlocks (embedVec v) (embedVec w) := by
  have hModW : w.size % d = 0 := by
    simpa [hSize] using hMod
  have hAddSize : (vecAdd v w).size = v.size := vecAdd_size_of_eq hSize
  have hAddMod : (vecAdd v w).size % d = 0 := by
    simpa [hAddSize] using hMod
  have hDiv : v.size / d = w.size / d := by
    simp [hSize]
  have hChunkSize : (chunkExact v d).size = (chunkExact w d).size := by
    simp [chunkExact_size, d_ne_zero, hDiv]
  unfold embedVec vecAddBlocks
  simp [hMod, hModW, hAddMod, hChunkSize]
  apply Array.ext
  · simp [chunkExact_size, d_ne_zero, hAddSize, hMod]
  · intro i hiL hiR
    have hExtract :
      (vecAdd v w).extract (i * d) (i * d + d) =
        vecAdd (v.extract (i * d) (i * d + d))
          (w.extract (i * d) (i * d + d)) := by
      exact vecAdd_extract_of_size_eq hSize _ _
    simp [chunkExact, d_ne_zero, embedElem, hExtract]

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

end SuperNeo
