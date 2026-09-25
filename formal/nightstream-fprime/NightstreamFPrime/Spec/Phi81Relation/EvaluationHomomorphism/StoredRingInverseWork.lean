import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverse
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
Counted stored polynomial normalization for the executed inverse.
The scan refines CompPoly Data/Array/Lemmas.lean and Raw/Core.lean at
050f0bc7e9780703beb8d178ec533e52bd87d649 (Apache-2.0; copyright 2025
CompPoly, authors Quang Dao and Gregor Mitscha-Baude). The prefix loop
refines Lean 4.30 Init/Prelude.lean Array.extract. Counts include allocation
and a conservative shared-buffer copy cost; unique buffers are not assumed.
This is the input-normalization part, not a complete inverse work bound.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverseWork

open NightstreamFPrime.Spec
open CompPoly (CPolynomial)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

private abbrev Base := ZMod goldilocksModulus
abbrev Raw := CPolynomial.Raw Base

private def scan (values : Raw) : (count : Nat) → count ≤ values.size →
    Result (Option (Fin values.size))
  | 0, _ => ⟨none, 3⟩
  | count + 1, bounded =>
      if values[count]'(by omega) != 0 then
        ⟨some ⟨count, by omega⟩, 8⟩
      else
        let prior := scan values count (by omega)
        ⟨prior.value, prior.work + 6⟩

private theorem scan_value (values : Raw) (count : Nat) (bounded : count ≤ values.size) :
    (scan values count bounded).value =
      Array.findIdxRev?.find (fun value : Base => value != 0) values
        ⟨count, Nat.lt_succ_of_le bounded⟩ := by
  induction count with
  | zero =>
      have zero : (⟨0, Nat.lt_succ_of_le bounded⟩ : Fin (values.size + 1)) = 0 := by
        apply Fin.ext
        simp
      rw [zero, Array.findIdxRev?.find.eq_def]
      rfl
  | succ count ih =>
      simp only [scan, Array.findIdxRev?.find]
      split
      · rfl
      · exact ih (by omega)

private theorem scan_work_le (values : Raw) (count : Nat) (bounded : count ≤ values.size) :
    (scan values count bounded).work ≤ 6 * count + 3 := by
  induction count with
  | zero => simp [scan]
  | succ count ih =>
      have prior := ih (by omega)
      simp only [scan]
      split <;> dsimp only <;> omega

private def lastNonzero (values : Raw) : Result (Option (Fin values.size)) :=
  let found := scan values values.size (Nat.le_refl _)
  ⟨found.value, found.work + 2⟩

private theorem lastNonzero_value (values : Raw) :
    (lastNonzero values).value = CPolynomial.Raw.lastNonzero values :=
  scan_value values values.size (Nat.le_refl _)

private theorem lastNonzero_work_le (values : Raw) :
    (lastNonzero values).work ≤ 6 * values.size + 5 := by
  have bounded := scan_work_le values values.size (Nat.le_refl _)
  change (scan values values.size _).work + 2 ≤ _
  omega

/-- The budget is the requested prefix length, not an input restriction.
Each push charges copying all previous cells and reserving the whole prefix.
The size-dependent charge is read before creating the next array. -/
private def copyLoop (values : Raw) (budget : Nat) : Nat → Nat → Raw → Result Raw
  | remaining, index, output =>
      if live : index < values.size then
        match remaining with
        | 0 => ⟨output, 3⟩
        | count + 1 =>
            let stepWork := 2 * output.size + budget + 8
            let next := copyLoop values budget count (index + 1) (output.push values[index])
            ⟨next.value, next.work + stepWork⟩
      else ⟨output, 2⟩

private theorem copyLoop_value (values : Raw) (budget remaining index : Nat) (output : Raw) :
    (copyLoop values budget remaining index output).value =
      Array.extract.loop values remaining index output := by
  induction remaining generalizing index output with
  | zero =>
      by_cases live : index < values.size
      · rw [copyLoop, Array.extract.loop, dif_pos live, dif_pos live]
      · rw [copyLoop, Array.extract.loop, dif_neg live, dif_neg live]
  | succ remaining ih =>
      by_cases live : index < values.size
      · rw [copyLoop, Array.extract.loop, dif_pos live, dif_pos live]
        exact ih (index + 1) (output.push values[index])
      · rw [copyLoop, Array.extract.loop, dif_neg live, dif_neg live]

private theorem copyLoop_work_le (values : Raw) (budget remaining index : Nat) (output : Raw)
    (fits : output.size + remaining ≤ budget) :
    (copyLoop values budget remaining index output).work ≤ remaining * (3 * budget + 8) + 3 := by
  induction remaining generalizing index output with
  | zero => rw [copyLoop]; split <;> dsimp only <;> omega
  | succ remaining ih =>
      rw [copyLoop]
      split
      · have next := ih (index + 1) (output.push values[index]) (by simp; omega)
        dsimp only
        have sizeBound : output.size ≤ budget := by omega
        nlinarith
      · dsimp only
        omega

private def copyPrefix (values : Raw) (count : Nat) : Result Raw :=
  let copied := copyLoop values count count 0 (Array.emptyWithCapacity count)
  ⟨copied.value, copied.work + count + 2⟩

private theorem copyPrefix_value (values : Raw) (count : Nat) (bounded : count ≤ values.size) :
    (copyPrefix values count).value = values.extract 0 count := by
  change (copyLoop values count count 0 (Array.emptyWithCapacity count)).value = _
  rw [copyLoop_value]
  simp only [Array.extract, Nat.min_eq_left bounded, Nat.sub_zero]
  rfl

private theorem copyPrefix_work_le (values : Raw) (count : Nat) :
    (copyPrefix values count).work ≤ count * (3 * count + 8) + count + 5 := by
  have copied := copyLoop_work_le values count count 0 (Array.emptyWithCapacity count) (by simp)
  change (copyLoop values count count 0 (Array.emptyWithCapacity count)).work + count + 2 ≤ _
  omega

/-- Normalize the actual input array, returning its value and work together. -/
def trim (values : Raw) : Result Raw :=
  let found := lastNonzero values
  match found.value with
  | none => ⟨#[], found.work + 3⟩
  | some index =>
      let copied := copyPrefix values (index.val + 1)
      ⟨copied.value, found.work + copied.work + 4⟩

theorem trim_value (values : Raw) : (trim values).value = CPolynomial.Raw.trim values := by
  unfold trim CPolynomial.Raw.trim
  dsimp only
  rw [← lastNonzero_value]
  cases found : (lastNonzero values).value with
  | none => rfl
  | some index =>
      exact copyPrefix_value values (index.val + 1) (by have := index.isLt; omega)

theorem trim_work_le (values : Raw) :
    (trim values).work ≤ values.size * (3 * values.size + 8) + 7 * values.size + 14 := by
  have search := lastNonzero_work_le values
  unfold trim
  dsimp only
  cases found : (lastNonzero values).value with
  | none => dsimp only; nlinarith
  | some index =>
      have copied := copyPrefix_work_le values (index.val + 1)
      have bound : index.val + 1 ≤ values.size := by have := index.isLt; omega
      have square := Nat.mul_self_le_mul_self bound
      dsimp only
      nlinarith

/-- The canonical constructor consumes the same counted trim output. -/
def encode (value : StoredRingInverse.StoredRing) : Result (CPolynomial Base) :=
  let normalized := trim value.toArray
  ⟨⟨normalized.value, by
      rw [trim_value]
      exact CPolynomial.Raw.Trim.isCanonical_trim value.toArray⟩,
    normalized.work + 3⟩

theorem encode_value (value : StoredRingInverse.StoredRing) :
    (encode value).value = StoredRingInverse.encode value := by
  apply Subtype.ext
  exact trim_value value.toArray

theorem encode_work_le (value : StoredRingInverse.StoredRing) :
    (encode value).work ≤ ringDegree * (3 * ringDegree + 8) + 7 * ringDegree + 17 := by
  have bounded := trim_work_le value.toArray
  simp only [Vector.size_toArray] at bounded
  change (trim value.toArray).work + 3 ≤ _
  omega

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverseWork
