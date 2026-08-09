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

private theorem foldlDecoderTerms_from
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (terms : List DecoderFieldTerm) (initial : GateField) :
    terms.foldl (fun value term =>
        value + term.2 * decoderAtomFieldValue source coordinates term.1)
        initial =
      initial + evalDecoderFieldTerms source coordinates terms := by
  induction terms generalizing initial with
  | nil =>
      exact (gateField_add_zero initial).symm
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      change (initial +
            head.2 * decoderAtomFieldValue source coordinates head.1) +
          evalDecoderFieldTerms source coordinates tail =
        initial +
          tail.foldl (fun value term =>
              value + term.2 *
                decoderAtomFieldValue source coordinates term.1)
            (0 + head.2 *
              decoderAtomFieldValue source coordinates head.1)
      rw [inductionHypothesis, gateField_zero_add, gateField_add_assoc]

private theorem evalDecoderFieldTerms_cons
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (head : DecoderFieldTerm) (tail : List DecoderFieldTerm) :
    evalDecoderFieldTerms source coordinates (head :: tail) =
      head.2 * decoderAtomFieldValue source coordinates head.1 +
        evalDecoderFieldTerms source coordinates tail := by
  change tail.foldl (fun value term =>
        value + term.2 * decoderAtomFieldValue source coordinates term.1)
      (0 + head.2 * decoderAtomFieldValue source coordinates head.1) = _
  rw [foldlDecoderTerms_from, gateField_zero_add]

private theorem evalDecoderFieldTerms_append
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (left right : List DecoderFieldTerm) :
    evalDecoderFieldTerms source coordinates (left ++ right) =
      evalDecoderFieldTerms source coordinates left +
        evalDecoderFieldTerms source coordinates right := by
  unfold evalDecoderFieldTerms
  rw [List.foldl_append]
  rw [foldlDecoderTerms_from]
  rfl

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

private def chunkFieldTerms : List DecoderFieldTerm :=
  [(.source (.chunkBit ⟨0, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 0)),
   (.source (.chunkBit ⟨1, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 1)),
   (.source (.chunkBit ⟨2, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 2)),
   (.source (.chunkBit ⟨3, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 3)),
   (.source (.chunkBit ⟨4, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 4)),
   (.source (.chunkBit ⟨5, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 5)),
   (.source (.chunkBit ⟨6, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 6)),
   (.source (.chunkBit ⟨7, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 7)),
   (.source (.chunkBit ⟨8, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 8)),
   (.source (.chunkBit ⟨9, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 9)),
   (.source (.chunkBit ⟨10, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 10)),
   (.source (.chunkBit ⟨11, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 11)),
   (.source (.chunkBit ⟨12, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 12)),
   (.source (.chunkBit ⟨13, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 13)),
   (.source (.chunkBit ⟨14, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 14)),
   (.source (.chunkBit ⟨15, by decide⟩),
      negOne * (highDenominatorInverse * twoPower 15))]

private def quotientLowFieldTerms : List DecoderFieldTerm :=
  [(.coordinate (.quotientLow 0),
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
      negOne * (highDenominatorInverse * (fieldResidue 5 * twoPower 12)))]

private def residueFieldTerms : List DecoderFieldTerm :=
  [(.coordinate .residueLeft, negOne * highDenominatorInverse),
   (.coordinate .residueRight, negOne * highDenominatorInverse)]

private def highTailFieldTerms : List DecoderFieldTerm :=
  chunkFieldTerms ++ quotientLowFieldTerms ++ residueFieldTerms

private theorem highConstantSplit :
    highDenominatorInverse * fieldResidue 65533 =
      highDenominatorInverse * fieldResidue 65535 +
        highDenominatorInverse * (negOne * fieldResidue 2) := by
  native_decide

/-- Independent coefficient-level expansion of
`high = (chunk - 5 * low - residueIndex) / (5 * 2^13)`.

Unlike the generated artifact, this list names coefficients as field
operations on the semantic denominator inverse. -/
def highFormulaFieldTerms : List DecoderFieldTerm :=
  (.source .one, highDenominatorInverse * fieldResidue 65533) ::
    highTailFieldTerms

/-- The same semantic atoms ordered as the independently stated derived-high
formula: chunk, low quotient, residue pair, then the constant offset. -/
def derivedHighFieldTerms : List DecoderFieldTerm :=
  (.source .one, highDenominatorInverse * fieldResidue 65535) ::
    (highTailFieldTerms ++
      [(.source .one,
        highDenominatorInverse * (negOne * fieldResidue 2))])

/-- Closed coefficient audit: the generated high LC is exactly the semantic
field expansion above, atom for atom and in the same order. -/
theorem generatedHighDecoder_fieldTerms_exact :
    generatedHighDecoderRhs.map (fun term =>
        (term.role, fieldResidue (coefficient term.coefficient))) =
      highFormulaFieldTerms := by
  native_decide

/-- Splitting the combined constant into `65535 - 2` preserves evaluation;
all source and coordinate terms stay unchanged. -/
theorem highFormulaFieldTerms_eval_derived
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (one : source SourceRole.one.column = 1) :
    evalDecoderFieldTerms source coordinates highFormulaFieldTerms =
      evalDecoderFieldTerms source coordinates derivedHighFieldTerms := by
  have sourceOne : fieldResidue (source 0) = 1 := by
    have rawOne : source 0 = 1 := by
      simpa [SourceRole.column] using one
    rw [rawOne]
    rfl
  unfold highFormulaFieldTerms derivedHighFieldTerms
  rw [evalDecoderFieldTerms_cons, evalDecoderFieldTerms_cons,
    evalDecoderFieldTerms_append, evalDecoderFieldTerms_cons]
  have empty : evalDecoderFieldTerms source coordinates [] = 0 := rfl
  rw [empty, gateField_add_zero]
  simp only [decoderAtomFieldValue, SourceRole.column, sourceOne,
    gateField_mul_one]
  let tailValue := evalDecoderFieldTerms source coordinates highTailFieldTerms
  change highDenominatorInverse * fieldResidue 65533 + tailValue =
    highDenominatorInverse * fieldResidue 65535 +
      (tailValue + highDenominatorInverse * (negOne * fieldResidue 2))
  rw [highConstantSplit]
  rw [gateField_add_assoc,
    gateField_add_comm
      (highDenominatorInverse * (negOne * fieldResidue 2)) tailValue]

private theorem rangeSixteen :
    List.range 16 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] := by
  decide

private theorem swapMultipliers
    (left right value : GateField) :
    left * (right * value) = right * (left * value) := by
  rw [← gateField_mul_assoc,
    gateField_mul_comm left right,
    gateField_mul_assoc]

private theorem chunkFieldTerms_value
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (one : source SourceRole.one.column = 1)
    (bits : Chunk.BitsBoolean source 0) :
    highDenominatorInverse * fieldResidue 65535 *
          decoderAtomFieldValue source coordinates (.source .one) +
        evalDecoderFieldTerms source coordinates chunkFieldTerms =
      highDenominatorInverse * fieldResidue (Chunk.chunkValue source 0) := by
  rw [← Chunk.chunkTerms_value one bits, fieldResidue_lcEval]
  have sourceOne : fieldResidue (source 0) = 1 := by
    have rawOne : source 0 = 1 := by
      simpa [SourceRole.column] using one
    rw [rawOne]
    rfl
  have c0 : fieldResidue (goldilocksP - 2 ^ 0) = negOne * twoPower 0 := by native_decide
  have c1 : fieldResidue (goldilocksP - 2 ^ 1) = negOne * twoPower 1 := by native_decide
  have c2 : fieldResidue (goldilocksP - 2 ^ 2) = negOne * twoPower 2 := by native_decide
  have c3 : fieldResidue (goldilocksP - 2 ^ 3) = negOne * twoPower 3 := by native_decide
  have c4 : fieldResidue (goldilocksP - 2 ^ 4) = negOne * twoPower 4 := by native_decide
  have c5 : fieldResidue (goldilocksP - 2 ^ 5) = negOne * twoPower 5 := by native_decide
  have c6 : fieldResidue (goldilocksP - 2 ^ 6) = negOne * twoPower 6 := by native_decide
  have c7 : fieldResidue (goldilocksP - 2 ^ 7) = negOne * twoPower 7 := by native_decide
  have c8 : fieldResidue (goldilocksP - 2 ^ 8) = negOne * twoPower 8 := by native_decide
  have c9 : fieldResidue (goldilocksP - 2 ^ 9) = negOne * twoPower 9 := by native_decide
  have c10 : fieldResidue (goldilocksP - 2 ^ 10) = negOne * twoPower 10 := by native_decide
  have c11 : fieldResidue (goldilocksP - 2 ^ 11) = negOne * twoPower 11 := by native_decide
  have c12 : fieldResidue (goldilocksP - 2 ^ 12) = negOne * twoPower 12 := by native_decide
  have c13 : fieldResidue (goldilocksP - 2 ^ 13) = negOne * twoPower 13 := by native_decide
  have c14 : fieldResidue (goldilocksP - 2 ^ 14) = negOne * twoPower 14 := by native_decide
  have c15 : fieldResidue (goldilocksP - 2 ^ 15) = negOne * twoPower 15 := by native_decide
  simp only [chunkFieldTerms, evalDecoderFieldTerms, ChunkRows.chunkTerms,
    rangeSixteen, List.map, List.foldl,
    decoderAtomFieldValue, SourceRole.column, ChunkRows.sourceBitCol,
    sourceOne, gateField_mul_one, gateField_zero_add]
  rw [List.foldl_append]
  simp only [List.foldl, gateField_zero_add]
  rw [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
  simp only [sourceOne, gateField_one_mul, gateField_mul_one,
    gateField_mul_add, gateField_mul_assoc]
  simp only [swapMultipliers negOne highDenominatorInverse,
    gateField_add_assoc]

private theorem quotientLowFieldTerms_value
    (source : SourceAssignment) (coordinates : CoordinateAssignment) :
    evalDecoderFieldTerms source coordinates quotientLowFieldTerms =
      negOne *
        (highDenominatorInverse *
          (fieldResidue 5 *
            (witnessOfCoordinates coordinates).quotientLowValue)) := by
  simp only [quotientLowFieldTerms, evalDecoderFieldTerms, List.foldl,
    decoderAtomFieldValue, CoordinateRole.column, witnessOfCoordinates,
    Witness.quotientLowValue, gateField_zero_add]
  simp only [gateField_mul_add, gateField_mul_assoc, gateField_mul_zero,
    gateField_add_zero]
  ac_rfl

private theorem residueFieldTerms_value
    (source : SourceAssignment) (coordinates : CoordinateAssignment) :
    evalDecoderFieldTerms source coordinates residueFieldTerms =
      negOne * highDenominatorInverse *
        ((witnessOfCoordinates coordinates).residueLeft +
          (witnessOfCoordinates coordinates).residueRight) := by
  simp only [residueFieldTerms, evalDecoderFieldTerms, List.foldl,
    decoderAtomFieldValue, CoordinateRole.column, witnessOfCoordinates,
    gateField_zero_add]
  rw [gateField_mul_add]

private theorem derivedHighAlgebraCore
    (inverse negative five two chunk low left right : GateField) :
    (((inverse * chunk + negative * (inverse * (five * low))) +
        negative * inverse * (left + right)) +
      inverse * (negative * two)) =
      inverse *
        ((chunk + negative * (five * low)) +
          negative * (left + right + two)) := by
  simp only [gateField_mul_add, gateField_mul_assoc]
  simp only [swapMultipliers negative inverse, gateField_add_assoc]

private theorem derivedHighAlgebra
    (chunk low left right : GateField) :
    (((highDenominatorInverse * chunk +
          negOne *
            (highDenominatorInverse * (fieldResidue 5 * low))) +
        negOne * highDenominatorInverse * (left + right)) +
      highDenominatorInverse * (negOne * fieldResidue 2)) =
      highDenominatorInverse *
        fieldSub (fieldSub chunk (fieldResidue 5 * low))
          (left + right + fieldResidue 2) := by
  unfold fieldSub negOne
  exact derivedHighAlgebraCore highDenominatorInverse
    (fieldResidue (goldilocksP - 1)) (fieldResidue 5) (fieldResidue 2)
    chunk low left right

/-- Evaluation of the generated high decoder LC is the independently stated
derived-high formula. This theorem is isolated to chunk zero, matching the
role-normalized artifact; it makes no production placement claim. -/
theorem generatedHighDecoderRhs_eq_derived
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (one : source SourceRole.one.column = 1)
    (bits : Chunk.BitsBoolean source 0) :
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
  rw [highFormulaFieldTerms_eval_derived source coordinates one]
  calc
    evalDecoderFieldTerms source coordinates derivedHighFieldTerms =
        (((highDenominatorInverse * fieldResidue 65535 *
              decoderAtomFieldValue source coordinates (.source .one) +
            evalDecoderFieldTerms source coordinates chunkFieldTerms) +
          evalDecoderFieldTerms source coordinates quotientLowFieldTerms) +
        evalDecoderFieldTerms source coordinates residueFieldTerms) +
          highDenominatorInverse * (negOne * fieldResidue 2) *
            decoderAtomFieldValue source coordinates (.source .one) := by
      unfold derivedHighFieldTerms highTailFieldTerms
      rw [evalDecoderFieldTerms_cons, evalDecoderFieldTerms_append,
        evalDecoderFieldTerms_append, evalDecoderFieldTerms_append,
        evalDecoderFieldTerms_cons]
      have empty : evalDecoderFieldTerms source coordinates [] = 0 := rfl
      rw [empty, gateField_add_zero]
      simp only [gateField_add_assoc, gateField_mul_assoc]
    _ =
        ((highDenominatorInverse * fieldResidue (Chunk.chunkValue source 0) +
            negOne *
              (highDenominatorInverse *
                (fieldResidue 5 *
                  (witnessOfCoordinates coordinates).quotientLowValue))) +
          negOne * highDenominatorInverse *
            ((witnessOfCoordinates coordinates).residueLeft +
              (witnessOfCoordinates coordinates).residueRight)) +
        highDenominatorInverse * (negOne * fieldResidue 2) := by
      rw [chunkFieldTerms_value source coordinates one bits,
        quotientLowFieldTerms_value, residueFieldTerms_value]
      simp only [decoderAtomFieldValue, SourceRole.column, sourceOne,
        gateField_mul_one]
    _ = derivedQuotientHigh source 0
          (witnessOfCoordinates coordinates) := by
      unfold derivedQuotientHigh Witness.residueIndex
      exact derivedHighAlgebra
        (fieldResidue (Chunk.chunkValue source 0))
        (witnessOfCoordinates coordinates).quotientLowValue
        (witnessOfCoordinates coordinates).residueLeft
        (witnessOfCoordinates coordinates).residueRight

/-- If the generated high decoder holds, its output source cell is exactly the
independently derived high quotient bit in the active Goldilocks carrier. -/
theorem generatedHighDecoder_output_eq_derived
    {source : SourceAssignment} {coordinates : CoordinateAssignment}
    (one : source SourceRole.one.column = 1)
    (bits : Chunk.BitsBoolean source 0)
    (holds : generatedHighDecoder.Holds source coordinates) :
    fieldResidue
        (source (SourceRole.quotientBit highSourceIndex).column) =
      derivedQuotientHigh source 0 (witnessOfCoordinates coordinates) := by
  rw [generatedHighDecoder_shape] at holds
  change source (SourceRole.quotientBit highSourceIndex).column =
      evalDecoderLinearCombination source coordinates
        generatedHighDecoderRhs at holds
  rw [holds]
  exact generatedHighDecoderRhs_eq_derived source coordinates one bits

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5
