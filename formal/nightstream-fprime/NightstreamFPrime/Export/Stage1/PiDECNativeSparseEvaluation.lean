import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation
import NightstreamFPrime.Export.NativePoseidon2RoundCore

/-! Native-word accumulation for the existing stored sparse form. Entry
order and reads are unchanged. Field arithmetic belongs to the existing
proved Goldilocks word operations. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECNativeSparseEvaluation

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.NativePoseidon2

private abbrev CanonicalWord := { word : UInt64 // word.toNat < goldilocksModulus }

@[inline] private def fromF (value : F) : CanonicalWord :=
  ⟨UInt64.ofNatLT value.val (Nat.lt_trans value.isLt (by decide)), by
    simpa only [UInt64.toNat_ofNatLT] using value.isLt⟩

@[inline] private def toF (value : CanonicalWord) : F :=
  ⟨value.val.toNat, value.property⟩

private theorem toF_fromF (value : F) : toF (fromF value) = value := by
  apply Fin.ext
  simp only [toF, fromF, UInt64.toNat_ofNatLT]

private theorem toF_denote (value : CanonicalWord) : toF value = value.val.denote := by
  apply Fin.ext
  change value.val.toNat = value.val.toNat % goldilocksModulus
  exact (Nat.mod_eq_of_lt value.property).symm

@[inline] private def addWord (left right : CanonicalWord) : CanonicalWord :=
  ⟨add64 left.val right.val, add64_canonical _ _ left.property right.property⟩

@[inline] private def mulWord (left right : CanonicalWord) : CanonicalWord :=
  ⟨mul64 left.val right.val, mul64_canonical _ _⟩

private theorem addWord_value (left right : CanonicalWord) :
    toF (addWord left right) = toF left + toF right := by
  rw [toF_denote, toF_denote left, toF_denote right]
  simpa only [addWord] using add64_denote left.val right.val left.property right.property

private theorem mulWord_value (left right : CanonicalWord) :
    toF (mulWord left right) = toF left * toF right := by
  rw [toF_denote, toF_denote left, toF_denote right]
  simpa only [mulWord] using mul64_denote left.val right.val left.property right.property

/-- Traverse the stored entries with a native-word accumulator. -/
@[inline] private def accumulate {columns : Nat} (read : Fin columns → F)
    (entries : List (SparseEntry columns)) (initial : CanonicalWord) : CanonicalWord :=
  entries.foldl (fun total entry =>
    addWord total (mulWord (fromF entry.coefficient) (fromF (read entry.column)))) initial

private theorem accumulate_value {columns : Nat} (read : Fin columns → F)
    (entries : List (SparseEntry columns)) (initial : CanonicalWord) :
    toF (accumulate read entries initial) =
      entries.foldl (fun total entry =>
        total + entry.coefficient * read entry.column) (toF initial) := by
  unfold accumulate
  induction entries generalizing initial with
  | nil => rfl
  | cons entry entries ih =>
      simp only [List.foldl_cons, ih,
        addWord_value, mulWord_value, toF_fromF]

/-- Evaluate the same sparse entries with a native field accumulator. -/
@[specialize] def nativeEvalSparse {columns : Nat} (form : SparseForm columns)
    (read : Fin columns → F) : F :=
  toF (accumulate read form.entries (fromF 0))

/-- Exact equality for every form and read, including empty forms, repeated
columns, zero coefficients and cancellation. No validity premise is required. -/
theorem nativeEvalSparse_eq_spec {columns : Nat} (form : SparseForm columns)
    (read : Fin columns → F) : nativeEvalSparse form read = form.evalSparse read := by
  unfold nativeEvalSparse SparseForm.evalSparse
  rw [accumulate_value, toF_fromF]

/-- Two native-word accumulators. -/
private structure PairWords where
  low : CanonicalWord
  high : CanonicalWord

/-- Traverse entries with two coefficients each: two native-word accumulators
and one read per entry. -/
@[inline] private def accumulatePair {columns : Nat} (read : Fin columns → F)
    (entries : List (Fin columns × F × F)) (initial : PairWords) : PairWords :=
  entries.foldl (fun state entry =>
    let value := fromF (read entry.1)
    ⟨addWord state.low (mulWord (fromF entry.2.1) value),
      addWord state.high (mulWord (fromF entry.2.2) value)⟩) initial

private theorem accumulatePair_value {columns : Nat} (read : Fin columns → F)
    (entries : List (Fin columns × F × F)) (initial : PairWords) :
    toF (accumulatePair read entries initial).low =
        entries.foldl (fun total entry => total + entry.2.1 * read entry.1) (toF initial.low) ∧
      toF (accumulatePair read entries initial).high =
        entries.foldl (fun total entry => total + entry.2.2 * read entry.1) (toF initial.high) := by
  unfold accumulatePair
  induction entries generalizing initial with
  | nil => exact ⟨rfl, rfl⟩
  | cons entry entries ih =>
      simp only [List.foldl_cons, ih, addWord_value, mulWord_value, toF_fromF, and_self]

/-- Evaluate two sparse forms over the same columns with one read per column. -/
@[specialize] def nativeEvalPair {columns : Nat} (entries : List (Fin columns × F × F))
    (read : Fin columns → F) : F × F :=
  let result := accumulatePair read entries ⟨fromF 0, fromF 0⟩
  (toF result.low, toF result.high)

/-- Each coordinate is the evaluation of its own sparse form, for every list of
entries and every read. -/
theorem nativeEvalPair_eq_spec {columns : Nat} (entries : List (Fin columns × F × F))
    (read : Fin columns → F) :
    nativeEvalPair entries read =
      ((SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.1⟩)).evalSparse read,
        (SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.2⟩)).evalSparse read) := by
  have value := accumulatePair_value read entries ⟨fromF 0, fromF 0⟩
  rw [toF_fromF] at value
  simp only [nativeEvalPair, SparseForm.evalSparse, List.foldl_map, value.1, value.2]

/-- Six native-word accumulators: three pairs. -/
private structure TripleWords where
  firstLow : CanonicalWord
  firstHigh : CanonicalWord
  secondLow : CanonicalWord
  secondHigh : CanonicalWord
  thirdLow : CanonicalWord
  thirdHigh : CanonicalWord

/-- Entries with three coefficient pairs: one read per entry for all of them. -/
@[inline] private def accumulateTriple {columns : Nat} (read : Fin columns → F)
    (entries : List (Fin columns × (F × F) × (F × F) × (F × F))) (initial : TripleWords) :
    TripleWords :=
  entries.foldl (fun state entry =>
    let value := fromF (read entry.1)
    ⟨addWord state.firstLow (mulWord (fromF entry.2.1.1) value),
      addWord state.firstHigh (mulWord (fromF entry.2.1.2) value),
      addWord state.secondLow (mulWord (fromF entry.2.2.1.1) value),
      addWord state.secondHigh (mulWord (fromF entry.2.2.1.2) value),
      addWord state.thirdLow (mulWord (fromF entry.2.2.2.1) value),
      addWord state.thirdHigh (mulWord (fromF entry.2.2.2.2) value)⟩)
    initial

private theorem accumulateTriple_value {columns : Nat} (read : Fin columns → F)
    (entries : List (Fin columns × (F × F) × (F × F) × (F × F))) (initial : TripleWords) :
    toF (accumulateTriple read entries initial).firstLow =
        entries.foldl (fun total entry => total + entry.2.1.1 * read entry.1)
          (toF initial.firstLow) ∧
      toF (accumulateTriple read entries initial).firstHigh =
        entries.foldl (fun total entry => total + entry.2.1.2 * read entry.1)
          (toF initial.firstHigh) ∧
      toF (accumulateTriple read entries initial).secondLow =
        entries.foldl (fun total entry => total + entry.2.2.1.1 * read entry.1)
          (toF initial.secondLow) ∧
      toF (accumulateTriple read entries initial).secondHigh =
        entries.foldl (fun total entry => total + entry.2.2.1.2 * read entry.1)
          (toF initial.secondHigh) ∧
      toF (accumulateTriple read entries initial).thirdLow =
        entries.foldl (fun total entry => total + entry.2.2.2.1 * read entry.1)
          (toF initial.thirdLow) ∧
      toF (accumulateTriple read entries initial).thirdHigh =
        entries.foldl (fun total entry => total + entry.2.2.2.2 * read entry.1)
          (toF initial.thirdHigh) := by
  unfold accumulateTriple
  induction entries generalizing initial with
  | nil => exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  | cons entry entries ih =>
      simp only [List.foldl_cons, ih, addWord_value, mulWord_value, toF_fromF, and_self]

/-- Evaluate three pairs of sparse forms over the same columns with one read per
column. -/
@[specialize] def nativeEvalTriple {columns : Nat}
    (entries : List (Fin columns × (F × F) × (F × F) × (F × F))) (read : Fin columns → F) :
    (F × F) × (F × F) × (F × F) :=
  let zero := fromF 0
  let result := accumulateTriple read entries ⟨zero, zero, zero, zero, zero, zero⟩
  ((toF result.firstLow, toF result.firstHigh),
    (toF result.secondLow, toF result.secondHigh),
    (toF result.thirdLow, toF result.thirdHigh))

/-- Each coordinate of each pair is the evaluation of its own sparse form, for
every list of entries and every read. -/
theorem nativeEvalTriple_eq_spec {columns : Nat}
    (entries : List (Fin columns × (F × F) × (F × F) × (F × F))) (read : Fin columns → F) :
    nativeEvalTriple entries read =
      (((SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.1.1⟩)).evalSparse read,
          (SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.1.2⟩)).evalSparse read),
        ((SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.2.1.1⟩)).evalSparse read,
          (SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.2.1.2⟩)).evalSparse read),
        ((SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.2.2.1⟩)).evalSparse read,
          (SparseForm.mk (entries.map fun entry => ⟨entry.1, entry.2.2.2.2⟩)).evalSparse read)) := by
  have value := accumulateTriple_value read entries
    ⟨fromF 0, fromF 0, fromF 0, fromF 0, fromF 0, fromF 0⟩
  simp only [toF_fromF] at value
  simp only [nativeEvalTriple, SparseForm.evalSparse, List.foldl_map, value.1, value.2.1,
    value.2.2.1, value.2.2.2.1, value.2.2.2.2.1, value.2.2.2.2.2]

/-- An entry of `nativeEvalTriple` whose six coefficients are canonical words. Converting
the entries once lets every read of the same entries skip the conversions and the nested pairs. -/
structure TripleEntry (columns : Nat) where
  column : Fin columns
  firstLow : { word : UInt64 // word.toNat < goldilocksModulus }
  firstHigh : { word : UInt64 // word.toNat < goldilocksModulus }
  secondLow : { word : UInt64 // word.toNat < goldilocksModulus }
  secondHigh : { word : UInt64 // word.toNat < goldilocksModulus }
  thirdLow : { word : UInt64 // word.toNat < goldilocksModulus }
  thirdHigh : { word : UInt64 // word.toNat < goldilocksModulus }

/-- The words of one entry of `nativeEvalTriple`. -/
def TripleEntry.ofEntry {columns : Nat} (entry : Fin columns × (F × F) × (F × F) × (F × F)) :
    TripleEntry columns :=
  ⟨entry.1, fromF entry.2.1.1, fromF entry.2.1.2, fromF entry.2.2.1.1, fromF entry.2.2.1.2,
    fromF entry.2.2.2.1, fromF entry.2.2.2.2⟩

/-- `nativeEvalTriple` of entries converted by `TripleEntry.ofEntry`. -/
@[specialize] def nativeEvalTripleWords {columns : Nat} (entries : Array (TripleEntry columns))
    (read : Fin columns → F) : (F × F) × (F × F) × (F × F) :=
  let zero := fromF 0
  let result := entries.foldl (fun (state : TripleWords) entry =>
    let value := fromF (read entry.column)
    ⟨addWord state.firstLow (mulWord entry.firstLow value),
      addWord state.firstHigh (mulWord entry.firstHigh value),
      addWord state.secondLow (mulWord entry.secondLow value),
      addWord state.secondHigh (mulWord entry.secondHigh value),
      addWord state.thirdLow (mulWord entry.thirdLow value),
      addWord state.thirdHigh (mulWord entry.thirdHigh value)⟩)
    ⟨zero, zero, zero, zero, zero, zero⟩
  ((toF result.firstLow, toF result.firstHigh),
    (toF result.secondLow, toF result.secondHigh),
    (toF result.thirdLow, toF result.thirdHigh))

/-- Converting the entries first does not change the evaluation. -/
theorem nativeEvalTripleWords_ofEntry {columns : Nat}
    (entries : List (Fin columns × (F × F) × (F × F) × (F × F))) (read : Fin columns → F) :
    nativeEvalTripleWords (entries.map TripleEntry.ofEntry).toArray read =
      nativeEvalTriple entries read := by
  simp only [nativeEvalTripleWords, nativeEvalTriple, accumulateTriple, List.foldl_toArray',
    List.foldl_map, TripleEntry.ofEntry]

end NightstreamFPrime.Export.Stage1.PiDECNativeSparseEvaluation
