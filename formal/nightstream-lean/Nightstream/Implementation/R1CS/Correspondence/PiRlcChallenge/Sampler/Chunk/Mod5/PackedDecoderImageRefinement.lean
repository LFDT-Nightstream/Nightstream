import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.PackedMod5
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedArtifactSchema
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedRows

/-!
Isolated decoder/image refinement for the packed Mod-5 high quotient bit.

Owns: the exact generated `decoderDefinitions[1]` projection; the alias
between source quotient bits `0..12` and the thirteen committed packed
coordinates; the coordinate witness image; and equality between the generated
high decoder and the independently stated derived-high formula.

Does not own: full-profile or full-F-prime placement, selectors, inactive
rows, the eight packed polynomial rows, `SevenNonresidue`, Rust trace
conformance, costs, physical row refinement, or row-removal authority.

Emits constraints: no.

Authority boundary: the generated decoder is non-authoritative data. Its
right-hand side is compared with `derivedQuotientHigh`, which is defined from
the independently interpreted source chunk and packed witness. Low-coordinate
aliases are explicit premises; they are not inferred from placement.

| Stage path | Generated object | Mathematical obligation | Assurance tier |
|---|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.decoder.low_image` | `CoordinateRole.quotientLow 0..12` | exact aliases to `SourceRole.quotientBit 0..12` | model-level |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.decoder.high_shape` | `decoderDefinitions[1]` | linear output is source quotient bit 13 | artifact-checked |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.decoder.high_value` | generated high LC | equals `derivedQuotientHigh` when the source one-column is one | artifact-checked, isolated profile |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5

open Nightstream.Implementation.R1CS
open PackedMod5Artifact
open PackedMod5ArtifactData

/-- Embed one of the thirteen committed low-bit indices into the fourteen-bit
source quotient carrier. -/
def lowSourceIndex (index : Fin 13) : Fin 14 :=
  ⟨index.val, Nat.lt_trans index.isLt (by decide)⟩

/-- The sole source quotient bit omitted from the committed coordinates. -/
def highSourceIndex : Fin 14 := ⟨13, by decide⟩

/-- Exact alias boundary between the source quotient cells and the committed
packed low coordinates. -/
def LowCoordinateAliases
    (source : SourceAssignment) (coordinates : CoordinateAssignment) : Prop :=
  ∀ index : Fin 13,
    coordinates (CoordinateRole.quotientLow index).column =
      source (SourceRole.quotientBit (lowSourceIndex index)).column

/-- Read the fifteen committed coordinates as the independent packed witness.
No high quotient bit is stored in this image. -/
def witnessOfCoordinates (coordinates : CoordinateAssignment) : Witness where
  quotientLow index :=
    fieldResidue (coordinates (CoordinateRole.quotientLow index).column)
  residueLeft := fieldResidue (coordinates CoordinateRole.residueLeft.column)
  residueRight := fieldResidue (coordinates CoordinateRole.residueRight.column)

/-- The coordinate image preserves every low quotient bit exactly in the
active Goldilocks carrier. -/
theorem witnessOfCoordinates_low_eq_source
    {source : SourceAssignment} {coordinates : CoordinateAssignment}
    (aliases : LowCoordinateAliases source coordinates)
    (index : Fin 13) :
    (witnessOfCoordinates coordinates).quotientLow index =
      fieldResidue
        (source (SourceRole.quotientBit (lowSourceIndex index)).column) := by
  change fieldResidue
      (coordinates (CoordinateRole.quotientLow index).column) = _
  rw [aliases index]

/-- Stable typed access to the generated high-bit decoder. -/
def generatedHighDecoder : DecoderDefinition :=
  decoderDefinitions.get ⟨1, by native_decide⟩

/-- Extract the generated high-bit right-hand side without duplicating the
artifact's coefficient list in handwritten code. -/
def generatedHighDecoderRhs : DecoderLinearCombination :=
  match generatedHighDecoder with
  | .linear _ rhs => rhs
  | .product _ _ _ => []

/-- The second generated decoder definition is exactly a linear definition of
source quotient bit thirteen. -/
theorem generatedHighDecoder_shape :
    generatedHighDecoder =
      .linear (.quotientBit highSourceIndex) generatedHighDecoderRhs := by
  native_decide

private theorem fieldResidue_mod (value : Nat) :
    fieldResidue (value % goldilocksP) = fieldResidue value := by
  apply Fin.ext
  simp [fieldResidue]

private theorem fieldResidue_zero : fieldResidue 0 = 0 := by
  rfl

private theorem fieldResidue_two : fieldResidue 2 = 2 := by
  native_decide

private theorem fieldResidue_foldl
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) (initial : Nat) :
    fieldResidue
        (terms.foldl (fun value term =>
          value + term.2 * assignment term.1) initial) =
      terms.foldl (fun value term =>
        value + fieldResidue term.2 * fieldResidue (assignment term.1))
        (fieldResidue initial) := by
  induction terms generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl]
      rw [inductionHypothesis, fieldResidue_add_hom, fieldResidue_mul_hom]

private theorem fieldResidue_chunkFold
    (assignment : Nat → Nat) (chunk : Nat) (offsets : List Nat)
    (initial : Nat) :
    fieldResidue
        (offsets.foldl (fun value offset =>
          value + 2 ^ offset *
            assignment (ChunkRows.sourceBitCol chunk offset)) initial) =
      offsets.foldl (fun value offset =>
        value + fieldResidue (2 ^ offset) *
          fieldResidue (assignment (ChunkRows.sourceBitCol chunk offset)))
        (fieldResidue initial) := by
  induction offsets generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl]
      rw [inductionHypothesis, fieldResidue_add_hom, fieldResidue_mul_hom]

private theorem fieldResidue_chunkValue
    (assignment : Nat → Nat) (chunk : Nat) :
    fieldResidue (Chunk.chunkValue assignment chunk) =
      (List.range 16).foldl (fun value offset =>
        value + fieldResidue (2 ^ offset) *
          fieldResidue (assignment (ChunkRows.sourceBitCol chunk offset))) 0 := by
  unfold Chunk.chunkValue
  rw [fieldResidue_chunkFold, fieldResidue_zero]

private theorem rangeSixteen :
    List.range 16 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] := by
  decide

private theorem fieldResidue_lcEval
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) :
    fieldResidue (lcEval assignment terms) =
      terms.foldl (fun value term =>
        value + fieldResidue term.2 * fieldResidue (assignment term.1)) 0 := by
  unfold lcEval
  rw [fieldResidue_mod, fieldResidue_foldl, fieldResidue_zero]

/-- Field value owned by one decoder atom before it is assigned a projected
decoder column. -/
def decoderAtomFieldValue
    (source : SourceAssignment) (coordinates : CoordinateAssignment) :
    DecoderAtom → GateField
  | .source role => fieldResidue (source role.column)
  | .coordinate role => fieldResidue (coordinates role.column)

private theorem sourceRole_column_lt_decoderOffset (role : SourceRole) :
    role.column < decoderCoordinateOffset := by
  cases role <;>
    simp [SourceRole.column, decoderCoordinateOffset,
      ChunkRows.sourceBitCol, ChunkRows.residueCol,
      ChunkRows.quotientCol, ChunkRows.residueProductCol,
      ChunkRows.quotientBitCol, ChunkRows.base] <;>
    omega

private theorem decoderAssignment_atom
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (atom : DecoderAtom) :
    fieldResidue
        (decoderAssignment source coordinates atom.column) =
      decoderAtomFieldValue source coordinates atom := by
  cases atom with
  | source role =>
      simp [decoderAssignment, decoderAtomFieldValue, DecoderAtom.column,
        sourceRole_column_lt_decoderOffset]
  | coordinate role =>
      have notLt :
          ¬ decoderCoordinateOffset + role.column < decoderCoordinateOffset := by
        omega
      simp [decoderAssignment, decoderAtomFieldValue, DecoderAtom.column,
        notLt]

abbrev DecoderFieldTerm := DecoderAtom × GateField

/-- Evaluate an already normalized role/field-coefficient decoder LC. -/
def evalDecoderFieldTerms
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (terms : List DecoderFieldTerm) : GateField :=
  terms.foldl (fun value term =>
    value + term.2 * decoderAtomFieldValue source coordinates term.1) 0

/-- Translating a role LC through the artifact column map and evaluating it
with the Nat R1CS evaluator agrees with direct field-role evaluation. -/
theorem fieldResidue_evalDecoderLinearCombination
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (terms : DecoderLinearCombination) :
    fieldResidue (evalDecoderLinearCombination source coordinates terms) =
      evalDecoderFieldTerms source coordinates
        (terms.map fun term =>
          (term.role, fieldResidue (coefficient term.coefficient))) := by
  rw [show evalDecoderLinearCombination source coordinates terms =
      lcEval (decoderAssignment source coordinates)
        (sparseTerms DecoderAtom.column terms) by rfl]
  rw [fieldResidue_lcEval]
  simp only [sparseTerms, evalDecoderFieldTerms, List.foldl_map]
  congr 1
  funext value term
  rw [decoderAssignment_atom]

private def negOne : GateField := fieldResidue (goldilocksP - 1)

private def twoPower (index : Nat) : GateField := fieldResidue (2 ^ index)

private def chunkIndex (value : Nat) (isLt : value < 16) : Fin 16 :=
  ⟨value, isLt⟩

/-- Independent coefficient-level expansion of
`high = (chunk - 5 * low - residueIndex) / (5 * 2^13)`.

Unlike the generated artifact, this list names coefficients as field
operations on the semantic denominator inverse. -/
def highFormulaFieldTerms : List DecoderFieldTerm :=
  [(.source .one,
      negOne * (highDenominatorInverse * fieldResidue 2)),
   (.source (.chunkBit 0), highDenominatorInverse * twoPower 0),
   (.source (.chunkBit 1), highDenominatorInverse * twoPower 1),
   (.source (.chunkBit 2), highDenominatorInverse * twoPower 2),
   (.source (.chunkBit 3), highDenominatorInverse * twoPower 3),
   (.source (.chunkBit 4), highDenominatorInverse * twoPower 4),
   (.source (.chunkBit 5), highDenominatorInverse * twoPower 5),
   (.source (.chunkBit 6), highDenominatorInverse * twoPower 6),
   (.source (.chunkBit 7), highDenominatorInverse * twoPower 7),
   (.source (.chunkBit 8), highDenominatorInverse * twoPower 8),
   (.source (.chunkBit 9), highDenominatorInverse * twoPower 9),
   (.source (.chunkBit 10), highDenominatorInverse * twoPower 10),
   (.source (.chunkBit 11), highDenominatorInverse * twoPower 11),
   (.source (.chunkBit 12), highDenominatorInverse * twoPower 12),
   (.source (.chunkBit 13), highDenominatorInverse * twoPower 13),
   (.source (.chunkBit 14), highDenominatorInverse * twoPower 14),
   (.source (.chunkBit 15), highDenominatorInverse * twoPower 15),
   (.coordinate (.quotientLow 0),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 0))),
   (.coordinate (.quotientLow 1),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 1))),
   (.coordinate (.quotientLow 2),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 2))),
   (.coordinate (.quotientLow 3),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 3))),
   (.coordinate (.quotientLow 4),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 4))),
   (.coordinate (.quotientLow 5),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 5))),
   (.coordinate (.quotientLow 6),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 6))),
   (.coordinate (.quotientLow 7),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 7))),
   (.coordinate (.quotientLow 8),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 8))),
   (.coordinate (.quotientLow 9),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 9))),
   (.coordinate (.quotientLow 10),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 10))),
   (.coordinate (.quotientLow 11),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 11))),
   (.coordinate (.quotientLow 12),
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 12))),
   (.coordinate .residueLeft, negOne * highDenominatorInverse),
   (.coordinate .residueRight, negOne * highDenominatorInverse)]

/-- The same semantic atoms ordered as the independently stated derived-high
formula: chunk, low quotient, residue pair, then the constant offset. -/
def derivedHighFieldTerms : List DecoderFieldTerm :=
  [(.source (.chunkBit (chunkIndex 0 (by decide))),
      highDenominatorInverse * twoPower 0),
   (.source (.chunkBit (chunkIndex 1 (by decide))),
      highDenominatorInverse * twoPower 1),
   (.source (.chunkBit (chunkIndex 2 (by decide))),
      highDenominatorInverse * twoPower 2),
   (.source (.chunkBit (chunkIndex 3 (by decide))),
      highDenominatorInverse * twoPower 3),
   (.source (.chunkBit (chunkIndex 4 (by decide))),
      highDenominatorInverse * twoPower 4),
   (.source (.chunkBit (chunkIndex 5 (by decide))),
      highDenominatorInverse * twoPower 5),
   (.source (.chunkBit (chunkIndex 6 (by decide))),
      highDenominatorInverse * twoPower 6),
   (.source (.chunkBit (chunkIndex 7 (by decide))),
      highDenominatorInverse * twoPower 7),
   (.source (.chunkBit (chunkIndex 8 (by decide))),
      highDenominatorInverse * twoPower 8),
   (.source (.chunkBit (chunkIndex 9 (by decide))),
      highDenominatorInverse * twoPower 9),
   (.source (.chunkBit (chunkIndex 10 (by decide))),
      highDenominatorInverse * twoPower 10),
   (.source (.chunkBit (chunkIndex 11 (by decide))),
      highDenominatorInverse * twoPower 11),
   (.source (.chunkBit (chunkIndex 12 (by decide))),
      highDenominatorInverse * twoPower 12),
   (.source (.chunkBit (chunkIndex 13 (by decide))),
      highDenominatorInverse * twoPower 13),
   (.source (.chunkBit (chunkIndex 14 (by decide))),
      highDenominatorInverse * twoPower 14),
   (.source (.chunkBit (chunkIndex 15 (by decide))),
      highDenominatorInverse * twoPower 15),
   (.coordinate (.quotientLow 0),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 0))),
   (.coordinate (.quotientLow 1),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 1))),
   (.coordinate (.quotientLow 2),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 2))),
   (.coordinate (.quotientLow 3),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 3))),
   (.coordinate (.quotientLow 4),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 4))),
   (.coordinate (.quotientLow 5),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 5))),
   (.coordinate (.quotientLow 6),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 6))),
   (.coordinate (.quotientLow 7),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 7))),
   (.coordinate (.quotientLow 8),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 8))),
   (.coordinate (.quotientLow 9),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 9))),
   (.coordinate (.quotientLow 10),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 10))),
   (.coordinate (.quotientLow 11),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 11))),
   (.coordinate (.quotientLow 12),
      highDenominatorInverse * (negOne * (fieldResidue 5 * twoPower 12))),
   (.coordinate .residueLeft, highDenominatorInverse * negOne),
   (.coordinate .residueRight, highDenominatorInverse * negOne),
   (.source .one, highDenominatorInverse * (negOne * fieldResidue 2))]

/-- Closed coefficient audit: the generated high LC is exactly the semantic
field expansion above, atom for atom and in the same order. -/
theorem generatedHighDecoder_fieldTerms_exact :
    generatedHighDecoderRhs.map (fun term =>
        (term.role, fieldResidue (coefficient term.coefficient))) =
      highFormulaFieldTerms := by
  native_decide

/-- Closed permutation audit between artifact order and semantic formula order. -/
theorem highFormulaFieldTerms_perm_derived :
    highFormulaFieldTerms.Perm derivedHighFieldTerms := by
  native_decide

private theorem evalDecoderFieldTerms_perm
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    {left right : List DecoderFieldTerm} (permutation : left.Perm right) :
    evalDecoderFieldTerms source coordinates left =
      evalDecoderFieldTerms source coordinates right := by
  unfold evalDecoderFieldTerms
  apply permutation.foldl_eq'
  intro leftTerm _ rightTerm _ value
  apply Fin.ext
  simp only [Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc,
    Nat.add_comm, Nat.add_left_comm]

/-- Evaluation of the generated high decoder LC is the independently stated
derived-high formula. This theorem is isolated to chunk zero, matching the
role-normalized artifact; it makes no production placement claim. -/
theorem generatedHighDecoderRhs_eq_derived
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (one : source SourceRole.one.column = 1) :
    fieldResidue
        (evalDecoderLinearCombination source coordinates
          generatedHighDecoderRhs) =
      derivedQuotientHigh source 0 (witnessOfCoordinates coordinates) := by
  have sourceOne : fieldResidue (source 0) = 1 := by
    have rawOne : source 0 = 1 := by
      simpa [SourceRole.column] using one
    rw [rawOne]
    rfl
  rw [fieldResidue_evalDecoderLinearCombination,
    generatedHighDecoder_fieldTerms_exact]
  rw [evalDecoderFieldTerms_perm source coordinates
    highFormulaFieldTerms_perm_derived]
  simp only [evalDecoderFieldTerms, derivedHighFieldTerms, chunkIndex,
    List.foldl, decoderAtomFieldValue, SourceRole.column,
    CoordinateRole.column, ChunkRows.sourceBitCol]
  simp only [derivedQuotientHigh, witnessOfCoordinates,
    Witness.quotientLowValue, Witness.residueIndex, fieldSub,
    twoPower, negOne]
  rw [fieldResidue_chunkValue, rangeSixteen]
  simp only [List.foldl, ChunkRows.sourceBitCol, CoordinateRole.column,
    Nat.pow_succ, Nat.pow_zero, Nat.reduceMul, Nat.reduceAdd]
  simp
  rw [sourceOne]
  simp only [gateField_mul_add, gateField_add_assoc, gateField_mul_assoc,
    gateField_mul_one, gateField_one_mul, fieldResidue_one,
    fieldResidue_two]

/-- If the generated high decoder holds, its output source cell is exactly the
independently derived high quotient bit in the active Goldilocks carrier. -/
theorem generatedHighDecoder_output_eq_derived
    {source : SourceAssignment} {coordinates : CoordinateAssignment}
    (one : source SourceRole.one.column = 1)
    (holds : generatedHighDecoder.Holds source coordinates) :
    fieldResidue
        (source (SourceRole.quotientBit highSourceIndex).column) =
      derivedQuotientHigh source 0 (witnessOfCoordinates coordinates) := by
  rw [generatedHighDecoder_shape] at holds
  change source (SourceRole.quotientBit highSourceIndex).column =
      evalDecoderLinearCombination source coordinates
        generatedHighDecoderRhs at holds
  rw [holds]
  exact generatedHighDecoderRhs_eq_derived source coordinates one

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5
