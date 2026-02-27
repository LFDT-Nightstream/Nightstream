import SuperNeo.CoeffMaps
import SuperNeo.Dimensions

/-! Vector/matrix embedding utilities for ring-to-field transport. -/


namespace SuperNeo

open F

/-- Element embedding: F^d -> R_F via coefficient map inverse. -/
def embedElem (v : Array F) : Coeffs := cfInv v

/-- Inverse element embedding: R_F -> F^d via coefficients. -/
def unembedElem (a : Coeffs) : Array F := cf a

private def chunkExact (xs : Array F) (chunk : Nat) : Array (Array F) :=
  Id.run do
    let mut out : Array (Array F) := #[]
    let mut i := 0
    while i < xs.size do
      let stop := Nat.min (i + chunk) xs.size
      out := out.push (xs.extract i stop)
      i := i + chunk
    return out

private def flatten (blocks : Array (Array F)) : Array F :=
  blocks.foldl (fun acc blk => acc ++ blk) #[]

/-- Vector embedding: F^(d*n_R) -> (R_F)^n_R by chunking in d coefficients. -/
def embedVec (z : Array F) : Array Coeffs :=
  if z.size % d != 0 then
    #[]
  else
    (chunkExact z d).map cfInv

def unembedVec (zr : Array Coeffs) : Array F :=
  flatten (zr.map cf)

/-- Matrix embedding row-wise. -/
def embedMatrix (m : Array (Array F)) : Array (Array Coeffs) :=
  m.map embedVec

def unembedMatrix (mr : Array (Array Coeffs)) : Array (Array F) :=
  mr.map unembedVec

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

def embeddingSanity : Bool :=
  let z := ((List.range (2 * d)).toArray).map F.ofNat
  embeddingVecRoundTrip z

end SuperNeo
