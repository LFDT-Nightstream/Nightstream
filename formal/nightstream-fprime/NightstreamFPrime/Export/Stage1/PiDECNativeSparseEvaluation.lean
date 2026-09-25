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

end NightstreamFPrime.Export.Stage1.PiDECNativeSparseEvaluation
