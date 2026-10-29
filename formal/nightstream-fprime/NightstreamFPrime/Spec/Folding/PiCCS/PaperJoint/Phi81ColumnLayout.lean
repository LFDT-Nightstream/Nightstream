import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

/-! SuperNeo v1.2 Section 5. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout

open NightstreamFPrime.Spec
open MatrixCoefficientSource

def blockCount (columns : Nat) : Nat :=
  (columns + ringDegree - 1) / ringDegree

@[implicit_reducible]
def flatIndex
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) : Nat :=
  block.val * ringDegree + coefficient.val

private theorem decodedBlock_lt
    {columns : Nat}
    (column : Fin columns) :
    column.val / ringDegree < blockCount columns := by
  simp [blockCount, ringDegree]
  omega

def decode
    {columns : Nat}
    (column : Fin columns) :
    Fin (blockCount columns) × Fin ringDegree :=
  (⟨column.val / ringDegree, decodedBlock_lt column⟩,
    ⟨column.val % ringDegree,
      Nat.mod_lt _ (by simp [ringDegree])⟩)

def encode?
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) : Option (Fin columns) :=
  if inLogicalRange : flatIndex block coefficient < columns then
    some ⟨flatIndex block coefficient, inLogicalRange⟩
  else
    none

theorem flatIndex_decode
    {columns : Nat}
    (column : Fin columns) :
    flatIndex (decode column).1 (decode column).2 = column.val := by
  simpa [flatIndex, decode, Nat.mul_comm] using
    Nat.div_add_mod column.val ringDegree

private theorem flatIndex_div
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) :
    flatIndex block coefficient / ringDegree = block.val := by
  simp [flatIndex, ringDegree]
  omega

private theorem flatIndex_mod
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) :
    flatIndex block coefficient % ringDegree = coefficient.val := by
  have bound : coefficient.val < 54 := coefficient.isLt
  simp [flatIndex, ringDegree, bound]

theorem decode_encode
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree)
    (column : Fin columns)
    (encoded : encode? block coefficient = some column) :
    decode column = (block, coefficient) := by
  unfold encode? at encoded
  split at encoded
  next inLogicalRange =>
    have exactColumn :
        (⟨flatIndex block coefficient, inLogicalRange⟩ : Fin columns) =
          column :=
      Option.some.inj encoded
    have valueEquality :
        flatIndex block coefficient = column.val :=
      congrArg Fin.val exactColumn
    apply Prod.ext
    · apply Fin.ext
      change column.val / ringDegree = block.val
      rw [← valueEquality]
      exact flatIndex_div block coefficient
    · apply Fin.ext
      change column.val % ringDegree = coefficient.val
      rw [← valueEquality]
      exact flatIndex_mod block coefficient
  next outsideLogicalRange =>
    contradiction

theorem encode_decode
    {columns : Nat}
    (column : Fin columns) :
    encode? (decode column).1 (decode column).2 = some column := by
  unfold encode?
  rw [flatIndex_decode]
  simp [column.isLt]

theorem encode_eq_none_iff
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) :
    encode? block coefficient = none ↔
      columns ≤ flatIndex block coefficient := by
  constructor
  · intro absent
    by_cases inLogicalRange : flatIndex block coefficient < columns
    · simp [encode?, inLogicalRange] at absent
    · omega
  · intro padding
    have outsideLogicalRange :
        ¬flatIndex block coefficient < columns := by
      exact Nat.not_lt.mpr padding
    simp [encode?, outsideLogicalRange]

theorem flatIndex_lt_paddedWidth
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) :
    flatIndex block coefficient < blockCount columns * ringDegree := by
  have blockBound := block.isLt
  have coefficientBound : coefficient.val < 54 := coefficient.isLt
  change
    block.val * 54 + coefficient.val < blockCount columns * 54
  omega

def layout (columns : Nat) :
    RingColumnLayout ringDegree (blockCount columns) columns where
  decode := decode
  encode? := encode?
  decode_encode := decode_encode
  encode_decode := encode_decode

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout
