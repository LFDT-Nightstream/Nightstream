import SuperNeo.Embedding

/-!
Definition-8 bar-lift layer.

This module now wires the concrete chunked path:

`embedVec -> map superneoBarBlock -> unembedVec`

for chunkable vectors (`size % d = 0`), and keeps the identity fallback for
non-chunkable vectors.
-/

namespace SuperNeo

open F

/-- Vector-level chunkability predicate for Definition-8 lifting. -/
def barLiftChunkableVec (v : Array F) : Prop :=
  v.size % d = 0

/-- Decidability of vector chunkability. -/
instance barLiftChunkableVec_decidable (v : Array F) :
    Decidable (barLiftChunkableVec v) := by
  unfold barLiftChunkableVec
  infer_instance

/-- Matrix-level chunkability predicate (row-wise). -/
def barLiftChunkableMatrix (m : Array (Array F)) : Prop :=
  ∀ i : Fin m.size, (m[i.1]'i.2).size % d = 0

/--
Vector bar-lift operator.

Chunkable vectors follow the blockwise lifting chain:
`unembedVec ((embedVec v).map (superneoBarBlock bar))`.
Non-chunkable vectors keep the identity fallback.
-/
def barLiftVector (bar : Array (Array F)) (v : Array F) : Array F :=
  if _hChunk : barLiftChunkableVec v then
    unembedVec ((embedVec v).map (superneoBarBlock bar))
  else
    v

/-- Matrix bar-lift operator (row-wise). -/
def barLiftMatrix (bar : Array (Array F)) (m : Array (Array F)) : Array (Array F) :=
  m.map (barLiftVector bar)

theorem barLiftVector_eq_barBlockRoundTrip_of_chunkable
    (bar : Array (Array F))
    (v : Array F)
    (hChunk : barLiftChunkableVec v) :
    barLiftVector bar v = unembedVec ((embedVec v).map (superneoBarBlock bar)) := by
  simp [barLiftVector, hChunk]

theorem barLiftVector_eq_self_of_not_chunkable
    (bar : Array (Array F))
    (v : Array F)
    (hNotChunk : ¬ barLiftChunkableVec v) :
    barLiftVector bar v = v := by
  simp [barLiftVector, hNotChunk]

/--
Compatibility bridge: if bar-block acts as identity on embedded blocks,
chunkable bar-lift equals embed/unembed round-trip.
-/
theorem barLiftVector_eq_embedRoundTrip_of_chunkable
    (bar : Array (Array F))
    (v : Array F)
    (hChunk : barLiftChunkableVec v)
    (hBlockId : ∀ blk : Coeffs, superneoBarBlock bar blk = blk) :
    barLiftVector bar v = unembedVec (embedVec v) := by
  have hPath := barLiftVector_eq_barBlockRoundTrip_of_chunkable bar v hChunk
  have hMapId : (embedVec v).map (superneoBarBlock bar) = embedVec v := by
    apply Array.ext
    · simp
    · intro i hiL hiR
      simp [hBlockId]
  simpa [hMapId] using hPath

@[simp] theorem barLiftVector_size (bar : Array (Array F)) (v : Array F) :
    (barLiftVector bar v).size = v.size := by
  by_cases hChunk : barLiftChunkableVec v
  · have hMod : v.size % d = 0 := by
      simpa [barLiftChunkableVec] using hChunk
    have hLiftPath :
        barLiftVector bar v = unembedVec ((embedVec v).map (superneoBarBlock bar)) := by
      exact barLiftVector_eq_barBlockRoundTrip_of_chunkable bar v hChunk
    have hRoundSize : (unembedVec (embedVec v)).size = v.size := by
      simpa using congrArg Array.size (unembedVec_embedVec_of_mod_eq_zero (z := v) hMod)
    calc
      (barLiftVector bar v).size
          = (unembedVec ((embedVec v).map (superneoBarBlock bar))).size := by
              simpa [hLiftPath]
      _ = (unembedVec (embedVec v)).size := by
            simp [unembedVec]
      _ = v.size := hRoundSize
  · simp [barLiftVector, hChunk]

@[simp] theorem barLiftMatrix_size (bar : Array (Array F)) (m : Array (Array F)) :
    (barLiftMatrix bar m).size = m.size := by
  simp [barLiftMatrix]

/--
Identity-style boundary for bar blocks.

This is the theorem-facing condition needed by identity-specialized closures.
-/
def barBlockIdentityAssumption (bar : Array (Array F)) : Prop :=
  ∀ blk : Coeffs, superneoBarBlock bar blk = blk

/--
Shape-driven closure for bar-block identity.

If the outer bar matrix does not have canonical ring dimension `d`, the
`superneoBarBlock` definition takes the fallback branch and returns inputs
unchanged.
-/
theorem barBlockIdentityAssumption_of_bar_size_ne_d
    (bar : Array (Array F))
    (hBarSize : bar.size ≠ d) :
    barBlockIdentityAssumption bar := by
  intro blk
  simp [barBlockIdentityAssumption, superneoBarBlock, hBarSize]

theorem barLiftVector_eq_of_barBlockIdentity
    (bar : Array (Array F))
    (v : Array F)
    (hId : barBlockIdentityAssumption bar) :
    barLiftVector bar v = v := by
  by_cases hChunk : barLiftChunkableVec v
  · have hMod : v.size % d = 0 := by
      simpa [barLiftChunkableVec] using hChunk
    calc
      barLiftVector bar v = unembedVec (embedVec v) := by
        exact barLiftVector_eq_embedRoundTrip_of_chunkable bar v hChunk hId
      _ = v := by
        exact unembedVec_embedVec_of_mod_eq_zero (z := v) hMod
  · exact barLiftVector_eq_self_of_not_chunkable bar v hChunk

theorem barLiftVector_eq
    (bar : Array (Array F))
    (v : Array F)
    (hId : barBlockIdentityAssumption bar) :
    barLiftVector bar v = v := by
  exact barLiftVector_eq_of_barBlockIdentity bar v hId

theorem barLiftMatrix_eq
    (bar : Array (Array F))
    (m : Array (Array F))
    (hId : barBlockIdentityAssumption bar) :
    barLiftMatrix bar m = m := by
  apply Array.ext
  · simp [barLiftMatrix]
  · intro i hiL hiR
    have hi : i < m.size := by simpa using hiR
    simpa [barLiftMatrix] using barLiftVector_eq bar (m[i]'hi) hId

private theorem map_superneoBarBlock_vecScaleBlocks_of_block_size_d
    (bar : Array (Array F))
    (s : F)
    (blocks : Array Coeffs)
    (hBlocks : ∀ i : Fin blocks.size, (blocks[i.1]'i.2).size = d) :
    (vecScaleBlocks s blocks).map (superneoBarBlock bar) =
      vecScaleBlocks s (blocks.map (superneoBarBlock bar)) := by
  apply Array.ext
  · simp [vecScaleBlocks]
  · intro i hiL hiR
    have hi : i < blocks.size := by simpa [vecScaleBlocks] using hiR
    have hBlkSize : (blocks[i]'hi).size = d := hBlocks ⟨i, hi⟩
    simpa [vecScaleBlocks, hi] using
      (superneoBarBlock_vecScale_of_size_d_ring
        (bar := bar) (x := blocks[i]'hi) (s := s) hBlkSize)

private theorem map_superneoBarBlock_vecAddBlocks_of_size_eq_of_block_size_d
    (bar : Array (Array F))
    (a b : Array Coeffs)
    (hSize : a.size = b.size)
    (ha : ∀ i : Fin a.size, (a[i.1]'i.2).size = d)
    (hb : ∀ i : Fin b.size, (b[i.1]'i.2).size = d) :
    (vecAddBlocks a b).map (superneoBarBlock bar) =
      vecAddBlocks (a.map (superneoBarBlock bar)) (b.map (superneoBarBlock bar)) := by
  apply Array.ext
  · simp [vecAddBlocks, hSize]
  · intro i hiL hiR
    have hi : i < a.size := by simpa [vecAddBlocks, hSize] using hiR
    have hiB : i < b.size := by simpa [hSize] using hi
    have hASize : (a[i]'hi).size = d := ha ⟨i, hi⟩
    have hBSize : (b[i]'hiB).size = d := hb ⟨i, hiB⟩
    simpa [vecAddBlocks, hSize, hi, hiB] using
      (superneoBarBlock_vecAdd_of_size_d_ring
        (bar := bar) (x := a[i]'hi) (y := b[i]'hiB) hASize hBSize)

theorem barLiftVector_add_constructive
    (bar : Array (Array F))
    (v w : Array F)
    (hSize : v.size = w.size) :
    barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w) := by
  by_cases hChunk : barLiftChunkableVec v
  · have hMod : v.size % d = 0 := by simpa [barLiftChunkableVec] using hChunk
    have hModW : w.size % d = 0 := by simpa [hSize] using hMod
    have hChunkW : barLiftChunkableVec w := by simpa [barLiftChunkableVec] using hModW
    have hAddSize : (vecAdd v w).size = v.size := vecAdd_size_of_eq hSize
    have hModAdd : (vecAdd v w).size % d = 0 := by simpa [hAddSize] using hMod
    have hChunkAdd : barLiftChunkableVec (vecAdd v w) := by
      simpa [barLiftChunkableVec] using hModAdd
    have hEmbedAdd :
        embedVec (vecAdd v w) = vecAddBlocks (embedVec v) (embedVec w) :=
      embedVec_vecAdd_of_size_mod_eq_zero hSize hMod
    have hEmbedSizeEq : (embedVec v).size = (embedVec w).size := by
      have hMulV : (embedVec v).size * d = v.size := by
        simpa using congrArg Array.size (unembedVec_embedVec_of_mod_eq_zero (z := v) hMod)
      have hMulW : (embedVec w).size * d = w.size := by
        simpa using congrArg Array.size (unembedVec_embedVec_of_mod_eq_zero (z := w) hModW)
      have hMulEq : (embedVec v).size * d = (embedVec w).size * d := by
        calc
          (embedVec v).size * d = v.size := hMulV
          _ = w.size := hSize
          _ = (embedVec w).size * d := hMulW.symm
      exact Nat.mul_right_cancel d_pos hMulEq
    have hBlockV : ∀ i : Fin (embedVec v).size, ((embedVec v)[i.1]'i.2).size = d := by
      intro i
      exact embedVec_block_size_of_mod_eq_zero hMod i
    have hBlockW : ∀ i : Fin (embedVec w).size, ((embedVec w)[i.1]'i.2).size = d := by
      intro i
      exact embedVec_block_size_of_mod_eq_zero hModW i
    have hMapAdd :
        (vecAddBlocks (embedVec v) (embedVec w)).map (superneoBarBlock bar) =
          vecAddBlocks ((embedVec v).map (superneoBarBlock bar))
            ((embedVec w).map (superneoBarBlock bar)) := by
      exact map_superneoBarBlock_vecAddBlocks_of_size_eq_of_block_size_d
        bar (embedVec v) (embedVec w) hEmbedSizeEq hBlockV hBlockW
    have hMapSizeEq :
        ((embedVec v).map (superneoBarBlock bar)).size =
          ((embedVec w).map (superneoBarBlock bar)).size := by
      simpa [hEmbedSizeEq]
    have hMapBlockV :
        ∀ i : Fin ((embedVec v).map (superneoBarBlock bar)).size,
          ((((embedVec v).map (superneoBarBlock bar))[i.1]'i.2).size = d) := by
      intro i
      have hi : i.1 < (embedVec v).size := by simpa using i.2
      have hBlk : ((embedVec v)[i.1]'hi).size = d := hBlockV ⟨i.1, hi⟩
      simpa [Array.getElem_map, hi] using
        (superneoBarBlock_size_of_size_d_ring
          (bar := bar) (a := (embedVec v)[i.1]'hi) hBlk)
    have hMapBlockW :
        ∀ i : Fin ((embedVec w).map (superneoBarBlock bar)).size,
          ((((embedVec w).map (superneoBarBlock bar))[i.1]'i.2).size = d) := by
      intro i
      have hi : i.1 < (embedVec w).size := by simpa using i.2
      have hBlk : ((embedVec w)[i.1]'hi).size = d := hBlockW ⟨i.1, hi⟩
      simpa [Array.getElem_map, hi] using
        (superneoBarBlock_size_of_size_d_ring
          (bar := bar) (a := (embedVec w)[i.1]'hi) hBlk)
    have hUnembedAdd :
        unembedVec
            (vecAddBlocks ((embedVec v).map (superneoBarBlock bar))
              ((embedVec w).map (superneoBarBlock bar))) =
          vecAdd
            (unembedVec ((embedVec v).map (superneoBarBlock bar)))
            (unembedVec ((embedVec w).map (superneoBarBlock bar))) := by
      exact unembedVec_vecAddBlocks_of_size_eq_of_block_size_d
        hMapSizeEq hMapBlockV hMapBlockW
    calc
      barLiftVector bar (vecAdd v w)
          = unembedVec ((embedVec (vecAdd v w)).map (superneoBarBlock bar)) := by
              exact barLiftVector_eq_barBlockRoundTrip_of_chunkable bar (vecAdd v w) hChunkAdd
      _ = unembedVec ((vecAddBlocks (embedVec v) (embedVec w)).map (superneoBarBlock bar)) := by
            simp [hEmbedAdd]
      _ = unembedVec
            (vecAddBlocks ((embedVec v).map (superneoBarBlock bar))
              ((embedVec w).map (superneoBarBlock bar))) := by
            simpa [hMapAdd]
      _ = vecAdd
            (unembedVec ((embedVec v).map (superneoBarBlock bar)))
            (unembedVec ((embedVec w).map (superneoBarBlock bar))) := hUnembedAdd
      _ = vecAdd (barLiftVector bar v) (barLiftVector bar w) := by
            simp [barLiftVector, hChunk, hChunkW]
  · have hModNe : v.size % d ≠ 0 := by
      simpa [barLiftChunkableVec] using hChunk
    have hModWNe : w.size % d ≠ 0 := by
      simpa [hSize] using hModNe
    have hChunkW : ¬ barLiftChunkableVec w := by
      simpa [barLiftChunkableVec] using hModWNe
    have hAddSize : (vecAdd v w).size = v.size := vecAdd_size_of_eq hSize
    have hChunkAdd : ¬ barLiftChunkableVec (vecAdd v w) := by
      intro h
      have hModAdd : (vecAdd v w).size % d = 0 := by simpa [barLiftChunkableVec] using h
      exact hModNe (by simpa [hAddSize] using hModAdd)
    simp [barLiftVector, hChunk, hChunkW, hChunkAdd]

theorem barLiftVector_scale_constructive
    (bar : Array (Array F))
    (s : F)
    (v : Array F) :
    barLiftVector bar (vecScale s v) = vecScale s (barLiftVector bar v) := by
  by_cases hChunk : barLiftChunkableVec v
  · have hMod : v.size % d = 0 := by simpa [barLiftChunkableVec] using hChunk
    have hScaleMod : (vecScale s v).size % d = 0 := by
      simpa [vecScale_size] using hMod
    have hChunkScale : barLiftChunkableVec (vecScale s v) := by
      simpa [barLiftChunkableVec] using hScaleMod
    have hEmbedScale :
        embedVec (vecScale s v) = vecScaleBlocks s (embedVec v) :=
      embedVec_vecScale_of_mod_eq_zero hMod s
    have hBlockV : ∀ i : Fin (embedVec v).size, ((embedVec v)[i.1]'i.2).size = d := by
      intro i
      exact embedVec_block_size_of_mod_eq_zero hMod i
    have hMapScale :
        (vecScaleBlocks s (embedVec v)).map (superneoBarBlock bar) =
          vecScaleBlocks s ((embedVec v).map (superneoBarBlock bar)) := by
      exact map_superneoBarBlock_vecScaleBlocks_of_block_size_d bar s (embedVec v) hBlockV
    calc
      barLiftVector bar (vecScale s v)
          = unembedVec ((embedVec (vecScale s v)).map (superneoBarBlock bar)) := by
              exact barLiftVector_eq_barBlockRoundTrip_of_chunkable bar (vecScale s v) hChunkScale
      _ = unembedVec ((vecScaleBlocks s (embedVec v)).map (superneoBarBlock bar)) := by
            simp [hEmbedScale]
      _ = unembedVec (vecScaleBlocks s ((embedVec v).map (superneoBarBlock bar))) := by
            simpa [hMapScale]
      _ = vecScale s (unembedVec ((embedVec v).map (superneoBarBlock bar))) := by
            exact unembedVec_vecScaleBlocks s ((embedVec v).map (superneoBarBlock bar))
      _ = vecScale s (barLiftVector bar v) := by
            simp [barLiftVector, hChunk]
  · have hModNe : v.size % d ≠ 0 := by
      simpa [barLiftChunkableVec] using hChunk
    have hScaleChunk : ¬ barLiftChunkableVec (vecScale s v) := by
      intro h
      have hModScale : (vecScale s v).size % d = 0 := by simpa [barLiftChunkableVec] using h
      exact hModNe (by simpa [vecScale_size] using hModScale)
    simp [barLiftVector, hChunk, hScaleChunk]

/--
Theorem-facing linearity contract for bar-lift.

Downstream theorem files should depend on this Prop boundary, not on Bool checks.
-/
def barLiftLinearityAssumption (bar : Array (Array F)) : Prop :=
  (∀ v w : Array F, v.size = w.size →
    barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w)) ∧
  (∀ s : F, ∀ v : Array F,
    barLiftVector bar (vecScale s v) = vecScale s (barLiftVector bar v))

theorem barLiftLinearityAssumption_constructive
    (bar : Array (Array F)) :
    barLiftLinearityAssumption bar := by
  constructor
  · intro v w hSize
    exact barLiftVector_add_constructive bar v w hSize
  · intro s v
    exact barLiftVector_scale_constructive bar s v

/--
If bar-block is identity on all coefficient blocks, bar-lift is linearly closed.
-/
theorem barLiftLinearityAssumption_of_barBlockIdentity
    (bar : Array (Array F))
    (hId : barBlockIdentityAssumption bar) :
    barLiftLinearityAssumption bar := by
  constructor
  · intro v w hSize
    have hVW : barLiftVector bar (vecAdd v w) = vecAdd v w :=
      barLiftVector_eq bar (vecAdd v w) hId
    have hV : barLiftVector bar v = v := barLiftVector_eq bar v hId
    have hW : barLiftVector bar w = w := barLiftVector_eq bar w hId
    calc
      barLiftVector bar (vecAdd v w)
          = vecAdd v w := hVW
      _ = vecAdd (barLiftVector bar v) (barLiftVector bar w) := by
            simpa [hV, hW]
  · intro s v
    have hSV : barLiftVector bar (vecScale s v) = vecScale s v :=
      barLiftVector_eq bar (vecScale s v) hId
    have hV : barLiftVector bar v = v := barLiftVector_eq bar v hId
    calc
      barLiftVector bar (vecScale s v)
          = vecScale s v := hSV
      _ = vecScale s (barLiftVector bar v) := by
            simpa [hV]

/--
Shape-fallback closure: if `bar.size ≠ d`, `superneoBarBlock` is identity and
bar-lift linearity follows constructively.
-/
theorem barLiftLinearityAssumption_of_bar_size_ne_d
    (bar : Array (Array F))
    (hBarSize : bar.size ≠ d) :
    barLiftLinearityAssumption bar := by
  exact barLiftLinearityAssumption_of_barBlockIdentity
    bar
    (barBlockIdentityAssumption_of_bar_size_ne_d bar hBarSize)

theorem barLiftVector_add
    (bar : Array (Array F))
    (v w : Array F)
    (hLift : barLiftLinearityAssumption bar)
    (hSize : v.size = w.size) :
    barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w) := by
  exact hLift.1 v w hSize

theorem barLiftVector_add_of_size_eq
    (bar : Array (Array F))
    (v w : Array F)
    (hLift : barLiftLinearityAssumption bar)
    (hSize : v.size = w.size) :
    barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w) := by
  exact hLift.1 v w hSize

theorem barLiftVector_scale
    (bar : Array (Array F))
    (s : F)
    (v : Array F)
    (hLift : barLiftLinearityAssumption bar) :
    barLiftVector bar (vecScale s v) = vecScale s (barLiftVector bar v) := by
  exact hLift.2 s v

/--
Check-facing universal contract for bar-lift linearity.

This is retained for executable compatibility, then bridged back to Prop.
-/
def barLiftLinearityCheckAssumption (bar : Array (Array F)) : Prop :=
  ∀ s : F, ∀ v w : Array F, v.size = w.size →
    decide (barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w)) = true ∧
    decide (barLiftVector bar (vecScale s v) = vecScale s (barLiftVector bar v)) = true

/-- Constructive closed theorem for bar-lift linearity (no boundary input). -/
theorem barLiftLinearityAssumption_closed
  (bar : Array (Array F)) :
  barLiftLinearityAssumption bar := by
  exact barLiftLinearityAssumption_constructive bar

/-- Identity wrapper for theorem-facing linearity (boundary-thread helper). -/
theorem barLiftLinearityAssumption_native
  (bar : Array (Array F))
  (_hLift : barLiftLinearityAssumption bar) :
  barLiftLinearityAssumption bar := by
  exact barLiftLinearityAssumption_closed bar

/-- Thread theorem-facing linearity through P9 context when supplied explicitly. -/
theorem barLiftLinearityAssumption_of_p9Embedding
  (bar : Array (Array F))
  (_hP9 : p9EmbeddingAssumption)
  (_hLift : barLiftLinearityAssumption bar) :
  barLiftLinearityAssumption bar := by
  exact barLiftLinearityAssumption_closed bar

/-- Closed wrapper is now explicit on theorem-facing linearity input. -/
theorem barLiftLinearityAssumption_of_p9Embedding_closed
  (bar : Array (Array F))
  (_hLift : barLiftLinearityAssumption bar) :
  barLiftLinearityAssumption bar := by
  exact barLiftLinearityAssumption_of_p9Embedding bar p9EmbeddingAssumption_holds
    (barLiftLinearityAssumption_closed bar)

/-- Convert theorem-facing bar-lift linearity into the check-facing contract. -/
theorem barLiftLinearityCheckAssumption_of_assumption
  {bar : Array (Array F)}
  (hAssm : barLiftLinearityAssumption bar) :
  barLiftLinearityCheckAssumption bar := by
  intro s v w hSize
  refine ⟨?_, ?_⟩
  · exact decide_eq_true (hAssm.1 v w hSize)
  · exact decide_eq_true (hAssm.2 s v)

/-- Convert check-facing bar-lift linearity into the theorem-facing contract. -/
theorem barLiftLinearityAssumption_of_checkAssumption
  {bar : Array (Array F)}
  (hCheck : barLiftLinearityCheckAssumption bar) :
  barLiftLinearityAssumption bar := by
  constructor
  · intro v w hSize
    exact decide_eq_true_eq.mp (hCheck 0 v w hSize).1
  · intro s v
    exact decide_eq_true_eq.mp (hCheck s v v rfl).2

/-- Equivalence between theorem-facing and check-facing bar-lift contracts. -/
theorem barLiftLinearityAssumption_iff_checkAssumption
  {bar : Array (Array F)} :
  barLiftLinearityAssumption bar ↔ barLiftLinearityCheckAssumption bar := by
  constructor
  · exact barLiftLinearityCheckAssumption_of_assumption
  · exact barLiftLinearityAssumption_of_checkAssumption

end SuperNeo
