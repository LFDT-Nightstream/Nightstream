import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.PackedMod5Schema
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.ChunkRows

/-!
Typed schema for the role-normalized packed Mod-5 artifact.

Owns: the finite source, coordinate, matrix, and row roles emitted by the
artifact generator; their explicit column maps; and direct Goldilocks
evaluators for source rows, decoder definitions, and sparse polynomial terms.

Does not own: generated production data, semantic packing proofs, selector
composition, transcript authority, or permission to remove source rows.

Emits constraints: no.

Authority boundary: generated lists are non-authoritative data. Their equations
must be evaluated and refined to `ChunkRows` before they can justify a claim.

| Stage path | Mathematical object | Direct evaluator |
|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.source` | role-normalized source R1CS row | `SourceRow.Holds` |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.decoder` | projected linear/product definition | `DecoderDefinition.Holds` |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.rows` | six low pairs, one high pair, one residue pair | `activeRowPoint` |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.polynomial` | sparse CCS residual | `evalPolynomial` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5Artifact

open Nightstream.Implementation.R1CS

/-- Map a role-normalized source block to the readable candidate-zero schema
owned by `ChunkRows`. -/
def SourceRole.column : SourceRole → Nat
  | .one => 0
  | .chunkBit offset => ChunkRows.sourceBitCol 0 offset
  | .index => ChunkRows.residueCol 0
  | .quotient => ChunkRows.quotientCol 0
  | .indexProduct stage => ChunkRows.residueProductCol 0 stage
  | .quotientBit offset => ChunkRows.quotientBitCol 0 offset

theorem SourceRole.column_injective : Function.Injective SourceRole.column := by
  intro left right equal
  cases left <;> cases right <;>
    simp [SourceRole.column, ChunkRows.sourceBitCol, ChunkRows.residueCol,
      ChunkRows.quotientCol, ChunkRows.residueProductCol,
      ChunkRows.quotientBitCol, ChunkRows.base] at equal ⊢ <;>
    omega

/-- Dense local coordinate index. -/
def CoordinateRole.column : CoordinateRole → Nat
  | .quotientLow index => index
  | .residueLeft => 13
  | .residueRight => 14

theorem CoordinateRole.column_injective :
    Function.Injective CoordinateRole.column := by
  intro left right equal
  cases left <;> cases right <;>
    simp [CoordinateRole.column] at equal ⊢ <;>
    omega

/-- Decoder coordinates occupy a disjoint interval immediately after the
candidate-zero source block. -/
def decoderCoordinateOffset : Nat := ChunkRows.base 1

/-- Collision-free decoder column map for the supported finite roles. -/
def DecoderAtom.column : DecoderAtom → Nat
  | .source role => role.column
  | .coordinate role => decoderCoordinateOffset + role.column

theorem DecoderAtom.column_injective : Function.Injective DecoderAtom.column := by
  intro left right equal
  cases left with
  | source leftRole =>
      cases right with
      | source rightRole =>
          congr 1
          exact SourceRole.column_injective equal
      | coordinate rightRole =>
          have leftLt : leftRole.column < decoderCoordinateOffset := by
            cases leftRole <;>
              simp [SourceRole.column, decoderCoordinateOffset,
                ChunkRows.sourceBitCol, ChunkRows.residueCol,
                ChunkRows.quotientCol, ChunkRows.residueProductCol,
                ChunkRows.quotientBitCol, ChunkRows.base] <;>
              omega
          simp [DecoderAtom.column] at equal
          omega
  | coordinate leftRole =>
      cases right with
      | source rightRole =>
          have rightLt : rightRole.column < decoderCoordinateOffset := by
            cases rightRole <;>
              simp [SourceRole.column, decoderCoordinateOffset,
                ChunkRows.sourceBitCol, ChunkRows.residueCol,
                ChunkRows.quotientCol, ChunkRows.residueProductCol,
                ChunkRows.quotientBitCol, ChunkRows.base] <;>
              omega
          simp [DecoderAtom.column] at equal
          omega
      | coordinate rightRole =>
          congr 1
          apply CoordinateRole.column_injective
          simp [DecoderAtom.column] at equal
          omega

/-- Exact production matrix index for each packed role. -/
def MatrixRole.index : MatrixRole → Nat
  | .selector => 0
  | .bitLeft => 44
  | .bitRight => 45
  | .residueLeft => 54
  | .residueRight => 55

theorem MatrixRole.index_injective : Function.Injective MatrixRole.index := by
  intro left right equal
  cases left <;> cases right <;>
    simp [MatrixRole.index] at equal ⊢

abbrev SourceAssignment := Nat → Nat
abbrev CoordinateAssignment := Nat → Nat
abbrev MatrixPoint := Nat → Nat

/-- Interpret a signed artifact coefficient as its canonical Goldilocks
residue without introducing a second field type. -/
def coefficient : Int → Nat
  | .ofNat value => value % goldilocksP
  | .negSucc predecessor =>
      let magnitude := (predecessor + 1) % goldilocksP
      if magnitude = 0 then 0 else goldilocksP - magnitude

/-- Translate a role-normalized LC into the active sparse R1CS carrier. -/
def sparseTerms {Role : Type} (column : Role → Nat)
    (terms : List (LinearTerm Role)) : List (Nat × Nat) :=
  terms.map fun term => (column term.role, coefficient term.coefficient)

/-- Evaluate a role-normalized LC with the active Goldilocks `lcEval`. -/
def evalLinearCombination {Role : Type} (column : Role → Nat)
    (assignment : Nat → Nat) (terms : List (LinearTerm Role)) : Nat :=
  lcEval assignment (sparseTerms column terms)

/-- Exact active `Row` denoted by one role-normalized source row. -/
def SourceRow.toRow (row : SourceRow) : Row :=
  ⟨sparseTerms SourceRole.column row.a,
    sparseTerms SourceRole.column row.b,
    sparseTerms SourceRole.column row.c⟩

/-- Direct source-row acceptance under the active R1CS semantics. -/
def SourceRow.Holds (row : SourceRow) (assignment : SourceAssignment) : Prop :=
  RowHolds assignment row.toRow

/-- Merge source and packed-coordinate assignments over the exported disjoint
decoder column map. -/
def decoderAssignment
    (source : SourceAssignment) (coordinates : CoordinateAssignment) : Nat → Nat :=
  fun column =>
    if column < decoderCoordinateOffset then source column
    else coordinates (column - decoderCoordinateOffset)

/-- Direct evaluation of one projected decoder LC. -/
def evalDecoderLinearCombination
    (source : SourceAssignment) (coordinates : CoordinateAssignment)
    (terms : DecoderLinearCombination) : Nat :=
  evalLinearCombination DecoderAtom.column
    (decoderAssignment source coordinates) terms

/-- Direct acceptance of one projected decoder definition. -/
def DecoderDefinition.Holds
    (definition : DecoderDefinition)
    (source : SourceAssignment) (coordinates : CoordinateAssignment) : Prop :=
  match definition with
  | .linear output rhs =>
      source output.column =
        evalDecoderLinearCombination source coordinates rhs
  | .product output left right =>
      source output.column =
        evalDecoderLinearCombination source coordinates left *
          evalDecoderLinearCombination source coordinates right % goldilocksP

/-- Direct sparse-monomial evaluation in active Goldilocks residues. -/
def evalPowers (point : MatrixPoint) : List VariablePower → Nat
  | [] => 1
  | factor :: tail =>
      point factor.role.index ^ factor.power * evalPowers point tail % goldilocksP

def evalPolynomialTerm (point : MatrixPoint) (term : PolynomialTerm) : Nat :=
  coefficient term.coefficient * evalPowers point term.powers % goldilocksP

def evalPolynomial (terms : List PolynomialTerm) (point : MatrixPoint) : Nat :=
  (terms.foldl
    (fun value term => value + evalPolynomialTerm point term) 0) % goldilocksP

/-- Read one packed bit operand from its local coordinates and the derived
high-bit decoder output. -/
def bitOperandValue
    (coordinates : CoordinateAssignment) (quotientHigh : Nat) : BitOperand → Nat
  | .quotientLow index => coordinates (CoordinateRole.quotientLow index).column
  | .quotientHigh => quotientHigh

/-- Sparse matrix point with exactly the five packed matrix roles populated. -/
def packedMatrixPoint
    (selector bitLeft bitRight residueLeft residueRight : Nat) : MatrixPoint :=
  fun matrix =>
    if matrix = MatrixRole.selector.index then selector
    else if matrix = MatrixRole.bitLeft.index then bitLeft
    else if matrix = MatrixRole.bitRight.index then bitRight
    else if matrix = MatrixRole.residueLeft.index then residueLeft
    else if matrix = MatrixRole.residueRight.index then residueRight
    else 0

/-- Matrix values placed on one active packed row. -/
def activeRowPoint
    (coordinates : CoordinateAssignment) (quotientHigh : Nat) : ActiveRow → MatrixPoint
  | .bitPair left right =>
      packedMatrixPoint 1
        (bitOperandValue coordinates quotientHigh left)
        (bitOperandValue coordinates quotientHigh right) 0 0
  | .residuePair =>
      packedMatrixPoint 1 0 0
        (coordinates CoordinateRole.residueLeft.column)
        (coordinates CoordinateRole.residueRight.column)

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5Artifact
