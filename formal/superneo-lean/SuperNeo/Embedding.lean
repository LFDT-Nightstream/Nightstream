import SuperNeo.CoeffMaps
import SuperNeo.Dimensions

/-! Vector/matrix embedding utilities for ring-to-field transport. -/


namespace SuperNeo

open F

/-- Element embedding: F^d -> R_F via coefficient map inverse. -/
def embedElem (v : Array F) : Coeffs := cfInv v

/-- Inverse element embedding: R_F -> F^d via coefficients. -/
def unembedElem (a : Coeffs) : Array F := cf a

theorem unembedElem_embedElem (v : Array F) :
  unembedElem (embedElem v) = v := by
  rfl

theorem embedElem_unembedElem (a : Coeffs) :
  embedElem (unembedElem a) = a := by
  rfl

theorem embedElem_vecAdd (v w : Array F) :
  embedElem (vecAdd v w) = vecAdd (embedElem v) (embedElem w) := by
  simpa [embedElem] using cfInv_vecAdd v w

theorem unembedElem_vecAdd (a b : Coeffs) :
  unembedElem (vecAdd a b) = vecAdd (unembedElem a) (unembedElem b) := by
  simpa [unembedElem] using cf_vecAdd a b

theorem embedElem_vecScale (s : F) (v : Array F) :
  embedElem (vecScale s v) = vecScale s (embedElem v) := by
  simpa [embedElem] using cfInv_vecScale s v

theorem unembedElem_vecScale (s : F) (a : Coeffs) :
  unembedElem (vecScale s a) = vecScale s (unembedElem a) := by
  simpa [unembedElem] using cf_vecScale s a

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
  if _hSize : a.size = b.size then
    Array.ofFn (fun i : Fin a.size =>
      vecAdd (a[i.1]'i.2) (b[i.1]'(by simpa [_hSize] using i.2)))
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

/-- Vector embedding: F^(d*n_R) -> (R_F)^n_R by chunking in d coefficients. -/
def embedVec (z : Array F) : Array Coeffs :=
  if z.size % d != 0 then
    #[]
  else
    (chunkExact z d).map cfInv

def unembedVec (zr : Array Coeffs) : Array F :=
  flatten (zr.map cf)

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
  simpa [vecScale] using
    (Array.map_extract (as := z) (f := fun x => s * x) (i := start) (j := stop)).symm

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
    simp [vecAdd, hSize, hiV, hiW]

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
  simp [hModScaled, hMod]
  apply Array.ext
  · simp [chunkExact_size, d_ne_zero, vecScale_size]
  · intro i hiL hiR
    simp [chunkExact, d_ne_zero, vecScale_extract]
    exact (embedElem_vecScale s
      (((z.extract (i * d) (i * d + d)) : Array F))).symm

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
    simpa [hSize]
  have hChunkSize : (chunkExact v d).size = (chunkExact w d).size := by
    simp [chunkExact_size, d_ne_zero, hDiv]
  unfold embedVec vecAddBlocks
  simp [hMod, hModW, hAddMod, hChunkSize]
  apply Array.ext
  · simp [chunkExact_size, d_ne_zero, hAddSize, hMod]
  · intro i hiL hiR
    have hBlkSize :
      (v.extract (i * d) (i * d + d)).size =
        (w.extract (i * d) (i * d + d)).size := by
      simp [hSize]
    have hExtract :
      (vecAdd v w).extract (i * d) (i * d + d) =
        vecAdd (v.extract (i * d) (i * d + d))
          (w.extract (i * d) (i * d + d)) := by
      exact vecAdd_extract_of_size_eq hSize _ _
    have hBlock :
      embedElem ((vecAdd v w).extract (i * d) (i * d + d)) =
        vecAdd (embedElem (v.extract (i * d) (i * d + d)))
          (embedElem (w.extract (i * d) (i * d + d))) := by
      simpa [hExtract] using
        (embedElem_vecAdd
          (v.extract (i * d) (i * d + d))
          (w.extract (i * d) (i * d + d)))
    simpa [chunkExact, d_ne_zero, hBlkSize, embedElem, hExtract] using hBlock

theorem embedVec_vecScale_of_mod
  {z : Array F}
  (hMod : z.size % d = 0)
  (s : F) :
  embedVec (vecScale s z) = vecScaleBlocks s (embedVec z) := by
  exact embedVec_vecScale_of_mod_eq_zero (z := z) hMod s

theorem embedVec_vecAdd_of_mod
  {v w : Array F}
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  embedVec (vecAdd v w) = vecAddBlocks (embedVec v) (embedVec w) := by
  exact embedVec_vecAdd_of_size_mod_eq_zero (v := v) (w := w) hSize hMod

/-- Matrix embedding row-wise. -/
def embedMatrix (m : Array (Array F)) : Array (Array Coeffs) :=
  m.map embedVec

def unembedMatrix (mr : Array (Array Coeffs)) : Array (Array F) :=
  mr.map unembedVec

/-- Row-wise matrix scaling on field vectors. -/
def matrixScaleRows (s : F) (m : Array (Array F)) : Array (Array F) :=
  m.map (vecScale s)

/-- Row-wise matrix scaling on embedded ring-vectors. -/
def matrixScaleRowsBlocks (s : F) (mr : Array (Array Coeffs)) : Array (Array Coeffs) :=
  mr.map (vecScaleBlocks s)

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

/-- Row-wise matrix addition on field vectors. -/
def matrixAddRows (m n : Array (Array F)) : Array (Array F) :=
  if _hSize : m.size = n.size then
    Array.ofFn (fun i : Fin m.size =>
      vecAdd (m[i.1]'i.2) (n[i.1]'(by simpa [_hSize] using i.2)))
  else
    #[]

/-- Row-wise matrix addition on embedded ring-vectors. -/
def matrixAddRowsBlocks (mr nr : Array (Array Coeffs)) : Array (Array Coeffs) :=
  if _hSize : mr.size = nr.size then
    Array.ofFn (fun i : Fin mr.size =>
      vecAddBlocks (mr[i.1]'i.2) (nr[i.1]'(by simpa [_hSize] using i.2)))
  else
    #[]

theorem embedMatrix_rowwise_vecAdd_of_rows_mod_eq_zero
  {m n : Array (Array F)}
  (hSize : m.size = n.size)
  (hRowsSize : ∀ i : Fin m.size, (m[i.1]'i.2).size = (n[i.1]'(by simpa [hSize] using i.2)).size)
  (hRowsMod : ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0) :
  embedMatrix (matrixAddRows m n) =
    matrixAddRowsBlocks (embedMatrix m) (embedMatrix n) := by
  unfold embedMatrix matrixAddRows matrixAddRowsBlocks
  simp [hSize]
  apply Array.ext
  · simp
  · intro i hiL hiR
    have hi : i < m.size := by
      simpa using hiL
    have hiN : i < n.size := by
      simpa [hSize] using hi
    have hRowSize : (m[i]'hi).size = (n[i]'hiN).size :=
      hRowsSize ⟨i, hi⟩
    have hRowMod : (m[i]'hi).size % d = 0 :=
      hRowsMod ⟨i, hi⟩
    have hMapM : (Array.map embedVec m)[i]! = embedVec (m[i]'hi) := by
      simp [hi]
    have hMapN : (Array.map embedVec n)[i]! = embedVec (n[i]'hiN) := by
      simp [hiN]
    have hRowSizeBang : (m[i]!).size = (n[i]!).size := by
      simpa [hi, hiN] using hRowSize
    have hRowModBang : (m[i]!).size % d = 0 := by
      simpa [hi] using hRowMod
    have hRowBang :
        embedVec (vecAdd (m[i]!) (n[i]!)) =
          vecAddBlocks (embedVec (m[i]!)) (embedVec (n[i]!)) :=
      embedVec_vecAdd_of_size_mod_eq_zero
        (v := m[i]!) (w := n[i]!) hRowSizeBang hRowModBang
    simpa [hi, hiN, hMapM, hMapN] using hRowBang

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

/-! ### Theorem-native P9 assumption interfaces -/

/-- Theorem-native vector embedding assumption (Definition 7 style). -/
def p9VecEmbeddingAssumption : Prop :=
  ∀ z : Array F, z.size % d = 0 → unembedVec (embedVec z) = z

/-- Check-style vector embedding assumption (kept for regression compatibility). -/
def p9VecEmbeddingCheckAssumption : Prop :=
  ∀ z : Array F, z.size % d = 0 → embeddingVecRoundTrip z = true

/-- Theorem-native matrix embedding assumption (row-wise Definition 7 style). -/
def p9MatrixEmbeddingAssumption : Prop :=
  ∀ m : Array (Array F),
    m.all (fun row => row.size % d = 0) = true →
      unembedMatrix (embedMatrix m) = m

/-- Check-style matrix embedding assumption (kept for regression compatibility). -/
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

theorem unembedVec_embedVec_eq_of_p9Assumption
  {z : Array F}
  (hAssm : p9VecEmbeddingAssumption)
  (hMod : z.size % d = 0) :
  unembedVec (embedVec z) = z := by
  exact hAssm z hMod

theorem unembedMatrix_embedMatrix_eq_of_p9Assumption
  {m : Array (Array F)}
  (hAssm : p9MatrixEmbeddingAssumption)
  (hRows : m.all (fun row => row.size % d = 0) = true) :
  unembedMatrix (embedMatrix m) = m := by
  exact hAssm m hRows

/-- Theorem-native element-level embedding bijection/linearity package. -/
def p9ElemEmbeddingAssumption : Prop :=
  (∀ v : Array F, unembedElem (embedElem v) = v) ∧
  (∀ a : Coeffs, embedElem (unembedElem a) = a) ∧
  (∀ v w : Array F, embedElem (vecAdd v w) = vecAdd (embedElem v) (embedElem w)) ∧
  (∀ s : F, ∀ v : Array F, embedElem (vecScale s v) = vecScale s (embedElem v))

theorem p9ElemEmbeddingAssumption_from_defs : p9ElemEmbeddingAssumption := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro v
    exact unembedElem_embedElem v
  · intro a
    exact embedElem_unembedElem a
  · intro v w
    exact embedElem_vecAdd v w
  · intro s v
    exact embedElem_vecScale s v

/-- Combined theorem-native P9 interface (element + vector + matrix). -/
def p9EmbeddingAssumption : Prop :=
  p9ElemEmbeddingAssumption ∧
    p9VecEmbeddingAssumption ∧
      p9MatrixEmbeddingAssumption

/-- Combined check-oriented P9 interface (vector + matrix checks). -/
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
  exact embeddingVecRoundTrip_true_of_p9VecAssumption (hAssm := p9EmbeddingAssumption_vec hAssm) hMod

theorem embeddingMatrixRoundTrip_true_of_p9EmbeddingAssumption
  {m : Array (Array F)}
  (hAssm : p9EmbeddingAssumption)
  (hRows : m.all (fun row => row.size % d = 0) = true) :
  embeddingMatrixRoundTrip m = true := by
  exact embeddingMatrixRoundTrip_true_of_p9MatrixAssumption
    (hAssm := p9EmbeddingAssumption_matrix hAssm) hRows

def embeddingSanity : Bool :=
  let z := ((List.range (2 * d)).toArray).map F.ofNat
  embeddingVecRoundTrip z

end SuperNeo
