import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/Phi81ColumnLayout.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Concrete logical-column to Phi81 block/lane layout.

Protocol: SuperNeo coefficient embedding (Section 5) and the concrete
`d = 54` assignment representation.
Phase: logical CCS column layout before coefficient-expanded matrix images.
Constraint family: flat column / ring block / coefficient lane / final padding.

Owns: the exact quotient/remainder ordering of a logical flat column inside
54-coefficient Phi81 blocks; the ceiling block count; a partial inverse from
padded block/lane positions to logical columns; and explicit recognition of
the zero-padding suffix of the final block.

Does not own: proof that Rust uses this layout, matrix-entry values, the Phi81
bar transform, transcript, SumCheck, R1CS lowering, row removal, or constraint
counts.

Emits constraints: no.

Authority boundary: logical columns are authoritative. A padded block/lane
position either decodes to exactly one logical column or is absent. An absent
position cannot carry a prover-selected field value; `MatrixCoefficientSource`
maps it to the additive identity.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | shape | block count | `blockCount = ceil(columns / 54)` |
| coefficient embedding | logical to packed | quotient / remainder | `decode(i) = (i / 54, i % 54)` |
| coefficient embedding | packed to logical | partial flattening | `encode?` returns `block * 54 + lane` exactly when logical |
| coefficient embedding | connectivity | round trip | logical columns and present padded positions are exact inverses |
| coefficient embedding | padding | final suffix | `encode? = none` iff the flat position is at least `columns` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout

open NightstreamFPrime.Spec
open MatrixCoefficientSource

/-- Number of 54-coefficient blocks needed to hold `columns` logical scalar
columns. This is the concrete ceiling-division spelling used by assignment
packing. -/
def blockCount (columns : Nat) : Nat :=
  (columns + ringDegree - 1) / ringDegree

/-- Flat logical index represented by one block/lane position. -/
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

/-- Quotient/remainder placement of one authoritative logical column. -/
def decode
    {columns : Nat}
    (column : Fin columns) :
    Fin (blockCount columns) × Fin ringDegree :=
  (⟨column.val / ringDegree, decodedBlock_lt column⟩,
    ⟨column.val % ringDegree,
      Nat.mod_lt _ (by simp [ringDegree])⟩)

/-- Partial inverse of `decode`. Positions in the final padding suffix return
`none` rather than a caller-controlled value. -/
def encode?
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) : Option (Fin columns) :=
  if inLogicalRange : flatIndex block coefficient < columns then
    some ⟨flatIndex block coefficient, inLogicalRange⟩
  else
    none

/-- Flattening quotient/remainder coordinates recovers the logical column. -/
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
  simp [flatIndex, ringDegree]

/-- Every present padded position decodes to the exact block/lane pair that
created it. -/
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

/-- Every logical column survives decode and re-encoding exactly. -/
theorem encode_decode
    {columns : Nat}
    (column : Fin columns) :
    encode? (decode column).1 (decode column).2 = some column := by
  unfold encode?
  rw [flatIndex_decode]
  simp [column.isLt]

/-- A padded position is absent exactly when its flat coordinate lies beyond
the authoritative logical column width. -/
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

/-- Every block/lane coordinate lies within the allocated padded width. -/
theorem flatIndex_lt_paddedWidth
    {columns : Nat}
    (block : Fin (blockCount columns))
    (coefficient : Fin ringDegree) :
    flatIndex block coefficient < blockCount columns * ringDegree := by
  have blockBound := block.isLt
  have coefficientBound : coefficient.val < 54 := by
    simpa [ringDegree] using coefficient.isLt
  change
    block.val * 54 + coefficient.val < blockCount columns * 54
  omega

/-- Concrete 54-lane partial layout consumed by the single-matrix source. -/
def layout (columns : Nat) :
    RingColumnLayout ringDegree (blockCount columns) columns where
  decode := decode
  encode? := encode?
  decode_encode := decode_encode
  encode_decode := encode_decode

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout
