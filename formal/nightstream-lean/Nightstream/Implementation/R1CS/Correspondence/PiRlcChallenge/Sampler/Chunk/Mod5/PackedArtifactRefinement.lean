import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.PackedMod5
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedArtifactSchema
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedRows

/-!
Artifact-checked refinement of the generated packed Mod-5 leaf.

Owns: exact generated shape checks, normalization of the twenty generated
source rows to the readable candidate-zero Mod-5 families, and interpretation
of the generated polynomial at independently populated role points.

Does not own: selectors, inactive rows, full-F′ placement, the Goldilocks
nonresidue certificate, Rust conformance, cost totals, source-to-coordinate
aliases, decoder-high evaluation, physical row refinement, or row-removal
authority.

Emits constraints: no.

Authority boundary: the generated payload is non-authoritative data. This file
evaluates it against independently named source and packed equations. The
polynomial identities are role-point identities only. Binding the thirteen low
coordinates to source quotient bits and the high role to `decoderDefinitions[1]`
remains open. The generated profile is the isolated one-rho, 64-chunk sampler
fixture; this file proves only its candidate-zero role-normalized leaf.

| Stage path | Generated object | Mathematical obligation | Assurance tier |
|---|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.source` | 20 source rows | residue range, 14 quotient bits, quotient recomposition, decomposition | artifact-checked |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.degree` | 12 sparse terms | exact selector-inclusive degrees and degree-eight ceiling | artifact-checked, leaf-local |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.bit_polynomial` | sparse bit-role polynomial | pair of Boolean residuals at an explicit role point | artifact-checked, leaf-local |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.residue_polynomial` | sparse residue-role polynomial | centered cubic plus centered-pair residual at an explicit role point | artifact-checked, leaf-local |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5

open Nightstream.Implementation.R1CS
open PackedMod5Artifact
open PackedMod5ArtifactData

/-- The readable candidate-zero source families represented by the generated
twenty-row payload. -/
def candidateZeroSourceRows : List Row :=
  ChunkRows.residueRangeRows 0 ++
    ChunkRows.quotientRangeRows 0 ++
    [ChunkRows.quotientRecompositionRow 0, ChunkRows.decompositionRow 0]

/-- Sparse-term order is not semantic. This is the exact normalization
relation used for the generated source rows. -/
def RowPermutation (left right : Row) : Prop :=
  left.a.Perm right.a ∧ left.b.Perm right.b ∧ left.c.Perm right.c

instance (left right : Row) : Decidable (RowPermutation left right) := by
  unfold RowPermutation
  infer_instance

inductive RowsPermutation : List Row → List Row → Prop where
  | nil : RowsPermutation [] []
  | cons {leftHead rightHead : Row} {leftTail rightTail : List Row}
      (head : RowPermutation leftHead rightHead)
      (tail : RowsPermutation leftTail rightTail) :
      RowsPermutation (leftHead :: leftTail) (rightHead :: rightTail)

private def rowsPermutationDecidable :
    (left right : List Row) → Decidable (RowsPermutation left right)
  | [], [] => isTrue .nil
  | [], _ :: _ => isFalse fun permutation => by cases permutation
  | _ :: _, [] => isFalse fun permutation => by cases permutation
  | leftHead :: leftTail, rightHead :: rightTail =>
      if head : RowPermutation leftHead rightHead then
        match rowsPermutationDecidable leftTail rightTail with
        | isTrue tail => isTrue (.cons head tail)
        | isFalse notTail => isFalse fun permutation => by
            cases permutation with
            | cons _ actualTail => exact notTail actualTail
      else
        isFalse fun permutation => by
          cases permutation with
          | cons actualHead _ => exact head actualHead

instance (left right : List Row) : Decidable (RowsPermutation left right) :=
  rowsPermutationDecidable left right

theorem rowHolds_iff_of_permutation
    (assignment : Nat → Nat) {left right : Row}
    (permutation : RowPermutation left right) :
    RowHolds assignment left ↔ RowHolds assignment right := by
  unfold RowHolds
  rw [Program.lcEval_eq_of_perm assignment permutation.1,
    Program.lcEval_eq_of_perm assignment permutation.2.1,
    Program.lcEval_eq_of_perm assignment permutation.2.2]

private theorem satisfies_cons_iff
    (head : Row) (tail : List Row) (assignment : Nat → Nat) :
    Satisfies (head :: tail) assignment ↔
      RowHolds assignment head ∧ Satisfies tail assignment := by
  simp [Satisfies]

theorem satisfies_iff_of_rowsPermutation
    (assignment : Nat → Nat) {left right : List Row}
    (permutation : RowsPermutation left right) :
    Satisfies left assignment ↔ Satisfies right assignment := by
  induction permutation with
  | nil => simp [Satisfies]
  | cons rowPermutation _tail inductionHypothesis =>
      rw [satisfies_cons_iff, satisfies_cons_iff,
        rowHolds_iff_of_permutation assignment rowPermutation,
        inductionHypothesis]

/-- Exact finite shape of the active generated payload. -/
theorem generated_shape_exact :
    schemaVersion = 1 ∧
      sourceInputOrder.length = 16 ∧
      sourceAllocatedOrder.length = 19 ∧
      sourceRows.length = 20 ∧
      coordinateOrder.length = 15 ∧
      decoderDefinitions.length = 6 ∧
      gateArity = 56 ∧
      matrixBindings =
        [{ role := .selector, index := 0 },
         { role := .bitLeft, index := 44 },
         { role := .bitRight, index := 45 },
         { role := .residueLeft, index := 54 },
         { role := .residueRight, index := 55 }] ∧
      (∀ binding ∈ matrixBindings, binding.index = binding.role.index) ∧
      activeRows.length = 8 ∧
      polynomialTerms.length = 12 := by
  native_decide

/-- Exact selector-inclusive degree of every generated sparse term. -/
theorem generated_polynomial_degrees_exact :
    polynomialTerms.map PolynomialTerm.totalDegree =
      [5, 4, 3, 5, 4, 3, 7, 5, 3, 5, 5, 5] := by
  native_decide

/-- The generated packed polynomial fits the production degree-eight ceiling.
Unlike manual degree bookkeeping, this statement ranges over the committed
artifact terms themselves. -/
theorem generated_polynomial_degree_at_most_eight :
    ∀ term ∈ polynomialTerms, term.totalDegree ≤ 8 := by
  native_decide

/-- The generator's role normalization produces the readable source row
families exactly up to semantically irrelevant sparse-term order. -/
theorem generated_source_rows_exact :
    RowsPermutation (sourceRows.map SourceRow.toRow) candidateZeroSourceRows := by
  native_decide

/-- Direct acceptance of every generated source row. -/
def GeneratedSourceAccepts (assignment : SourceAssignment) : Prop :=
  ∀ row ∈ sourceRows, row.Holds assignment

/-- Readable acceptance of the exact candidate-zero Mod-5 source subset. -/
def CandidateZeroSourceAccepts (assignment : SourceAssignment) : Prop :=
  Satisfies candidateZeroSourceRows assignment

/-- The generated twenty source rows normalize exactly to the active readable
candidate-zero equations. -/
theorem generatedSourceAccepts_iff_candidateZero
    (assignment : SourceAssignment) :
    GeneratedSourceAccepts assignment ↔ CandidateZeroSourceAccepts assignment := by
  constructor
  · intro generated
    apply (satisfies_iff_of_rowsPermutation assignment
      generated_source_rows_exact).mp
    intro row member
    rcases List.mem_map.mp member with ⟨sourceRow, sourceMember, rfl⟩
    exact generated sourceRow sourceMember
  · intro readable sourceRow sourceMember
    have normalized : Satisfies (sourceRows.map SourceRow.toRow) assignment :=
      (satisfies_iff_of_rowsPermutation assignment
        generated_source_rows_exact).mpr readable
    exact normalized sourceRow.toRow
      (List.mem_map.mpr ⟨sourceRow, sourceMember, rfl⟩)

private theorem coefficient_negTwo :
    coefficient (-2) = goldilocksP - 2 := by
  native_decide

private theorem coefficient_one : coefficient 1 = 1 := by
  native_decide

private theorem coefficient_fourteen : coefficient 14 = 14 := by
  native_decide

private theorem coefficient_negSeven :
    coefficient (-7) = goldilocksP - 7 := by
  native_decide

private theorem generated_bit_polynomial_expanded (left right : GateField) :
    fieldResidue
        (evalPolynomial polynomialTerms
          (packedMatrixPoint 1 left.val right.val 0 0)) =
      (((((left * left * left * left +
            fieldResidue (goldilocksP - 2) * (left * left * left)) +
          left * left) +
          fieldResidue (goldilocksP - 7) *
            (right * right * right * right)) +
          fieldResidue 14 * (right * right * right)) +
          fieldResidue (goldilocksP - 7) * (right * right)) := by
  apply Fin.ext
  simp [evalPolynomial, evalPolynomialTerm, evalPowers, polynomialTerms,
    packedMatrixPoint, MatrixRole.index, coefficient_negTwo,
    coefficient_negSeven, coefficient_one, coefficient_fourteen,
    fieldResidue, Fin.val_add, Fin.val_mul, Nat.pow_succ]
  simp only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod]

private theorem bitResidual_linear (value : GateField) :
    bitResidual value =
      value * value + fieldResidue (goldilocksP - 1) * value := by
  unfold bitResidual fieldSub
  rw [gateField_mul_add, gateField_mul_one,
    gateField_mul_comm value (fieldResidue (goldilocksP - 1))]

private theorem bitResidual_square_expansion (value : GateField) :
    bitResidual value * bitResidual value =
      (value * value * value * value +
          fieldResidue (goldilocksP - 2) * (value * value * value)) +
        value * value := by
  rw [bitResidual_linear, gateField_add_mul,
    gateField_mul_add, gateField_mul_add]
  calc
    _ = value * value * value * value +
          fieldResidue (goldilocksP - 1) * (value * value * value) +
          (fieldResidue (goldilocksP - 1) * (value * value * value) +
            (fieldResidue (goldilocksP - 1) *
                fieldResidue (goldilocksP - 1)) * (value * value)) := by
        ac_rfl
    _ = (value * value * value * value +
          (fieldResidue (goldilocksP - 1) * (value * value * value) +
            fieldResidue (goldilocksP - 1) * (value * value * value))) +
          value * value := by
      rw [negOne_mul_negOne, gateField_one_mul]
      apply gateField_add_reassociate_four
    _ = _ := by
      have combine (term : GateField) :
          fieldResidue (goldilocksP - 1) * term +
              fieldResidue (goldilocksP - 1) * term =
            fieldResidue (goldilocksP - 2) * term := by
        rw [← gateField_add_mul, negOne_add_negOne]
      rw [combine]

/-- The six generated bit terms are exactly the quadratic-nonresidue packing
of two Boolean residuals. -/
theorem generated_bit_polynomial (left right : GateField) :
    fieldResidue
        (evalPolynomial polynomialTerms
          (packedMatrixPoint 1 left.val right.val 0 0)) =
      bitResidual left * bitResidual left +
        fieldResidue (goldilocksP - 7) *
          (bitResidual right * bitResidual right) := by
  rw [generated_bit_polynomial_expanded,
    bitResidual_square_expansion, bitResidual_square_expansion,
    gateField_mul_add, gateField_mul_add]
  have middle :
      fieldResidue (goldilocksP - 7) *
          (fieldResidue (goldilocksP - 2) *
            (right * right * right)) =
        fieldResidue 14 * (right * right * right) := by
    rw [← gateField_mul_assoc, negSeven_mul_negTwo]
  rw [middle]
  apply Fin.ext
  simp only [Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem centeredResidual_linear (value : GateField) :
    centeredResidual value =
      value * value * value +
        fieldResidue (goldilocksP - 1) * value := by
  change bitResidual value * (value + 1) = _
  rw [bitResidual_linear, gateField_add_mul,
    gateField_mul_add, gateField_mul_add,
    gateField_mul_one, gateField_mul_one]
  have cancel :
      value * value +
          fieldResidue (goldilocksP - 1) * (value * value) = 0 := by
    rw [gateField_add_comm, negOne_mul_add_self]
  calc
    _ = (value * value * value +
          (value * value +
            fieldResidue (goldilocksP - 1) * (value * value))) +
          fieldResidue (goldilocksP - 1) * value := by
      apply Fin.ext
      simp only [Fin.val_add, Fin.val_mul]
      simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.mod_mul_mod,
        Nat.mul_mod_mod, Nat.add_assoc, Nat.mul_assoc, Nat.mul_comm]
    _ = _ := by
      rw [cancel, gateField_add_zero]

private theorem residuePairResidual_linear (witness : Witness) :
    witness.residuePairResidual =
      witness.residueRight * witness.residueLeft +
        fieldResidue (goldilocksP - 1) *
          (witness.residueRight * witness.residueRight) := by
  unfold Witness.residuePairResidual fieldSub
  rw [gateField_mul_add]
  congr 1
  ac_rfl

private theorem centeredResidual_square_expansion (value : GateField) :
    centeredResidual value * centeredResidual value =
      (value * value * value * value * value * value +
          fieldResidue (goldilocksP - 2) *
            (value * value * value * value)) +
        value * value := by
  rw [centeredResidual_linear, gateField_add_mul,
    gateField_mul_add, gateField_mul_add]
  calc
    _ = value * value * value * value * value * value +
          fieldResidue (goldilocksP - 1) *
              (value * value * value * value) +
          (fieldResidue (goldilocksP - 1) *
              (value * value * value * value) +
            (fieldResidue (goldilocksP - 1) *
                fieldResidue (goldilocksP - 1)) * (value * value)) := by
        ac_rfl
    _ = (value * value * value * value * value * value +
          (fieldResidue (goldilocksP - 1) *
              (value * value * value * value) +
            fieldResidue (goldilocksP - 1) *
              (value * value * value * value))) +
          value * value := by
      rw [negOne_mul_negOne, gateField_one_mul]
      apply gateField_add_reassociate_four
    _ = _ := by
      have combine (term : GateField) :
          fieldResidue (goldilocksP - 1) * term +
              fieldResidue (goldilocksP - 1) * term =
            fieldResidue (goldilocksP - 2) * term := by
        rw [← gateField_add_mul, negOne_add_negOne]
      rw [combine]

private theorem residuePairResidual_square_expansion (witness : Witness) :
    witness.residuePairResidual * witness.residuePairResidual =
      (witness.residueLeft * witness.residueLeft *
          witness.residueRight * witness.residueRight +
        fieldResidue (goldilocksP - 2) *
          (witness.residueLeft * witness.residueRight *
            witness.residueRight * witness.residueRight)) +
        witness.residueRight * witness.residueRight *
          witness.residueRight * witness.residueRight := by
  rw [residuePairResidual_linear, gateField_add_mul,
    gateField_mul_add, gateField_mul_add]
  calc
    _ = witness.residueLeft * witness.residueLeft *
            witness.residueRight * witness.residueRight +
          fieldResidue (goldilocksP - 1) *
              (witness.residueLeft * witness.residueRight *
                witness.residueRight * witness.residueRight) +
          (fieldResidue (goldilocksP - 1) *
              (witness.residueLeft * witness.residueRight *
                witness.residueRight * witness.residueRight) +
            (fieldResidue (goldilocksP - 1) *
                fieldResidue (goldilocksP - 1)) *
              (witness.residueRight * witness.residueRight *
                witness.residueRight * witness.residueRight)) := by
        ac_rfl
    _ = (witness.residueLeft * witness.residueLeft *
            witness.residueRight * witness.residueRight +
          (fieldResidue (goldilocksP - 1) *
              (witness.residueLeft * witness.residueRight *
                witness.residueRight * witness.residueRight) +
            fieldResidue (goldilocksP - 1) *
              (witness.residueLeft * witness.residueRight *
                witness.residueRight * witness.residueRight))) +
          witness.residueRight * witness.residueRight *
            witness.residueRight * witness.residueRight := by
      rw [negOne_mul_negOne, gateField_one_mul]
      apply gateField_add_reassociate_four
    _ = _ := by
      have combine (term : GateField) :
          fieldResidue (goldilocksP - 1) * term +
              fieldResidue (goldilocksP - 1) * term =
            fieldResidue (goldilocksP - 2) * term := by
        rw [← gateField_add_mul, negOne_add_negOne]
      rw [combine]

private theorem generated_residue_polynomial_expanded (witness : Witness) :
    fieldResidue
        (evalPolynomial polynomialTerms
          (packedMatrixPoint 1 0 0 witness.residueLeft.val
            witness.residueRight.val)) =
      (((((witness.residueLeft * witness.residueLeft *
                witness.residueLeft * witness.residueLeft *
                witness.residueLeft * witness.residueLeft +
            fieldResidue (goldilocksP - 2) *
              (witness.residueLeft * witness.residueLeft *
                witness.residueLeft * witness.residueLeft)) +
          witness.residueLeft * witness.residueLeft) +
          fieldResidue (goldilocksP - 7) *
            ((witness.residueLeft * witness.residueLeft) *
              (witness.residueRight * witness.residueRight))) +
          fieldResidue 14 *
            (witness.residueLeft *
              (witness.residueRight * witness.residueRight *
                witness.residueRight))) +
          fieldResidue (goldilocksP - 7) *
            (witness.residueRight * witness.residueRight *
              witness.residueRight * witness.residueRight)) := by
  apply Fin.ext
  simp [evalPolynomial, evalPolynomialTerm, evalPowers, polynomialTerms,
    packedMatrixPoint, MatrixRole.index, coefficient_negTwo,
    coefficient_negSeven, coefficient_one, coefficient_fourteen,
    fieldResidue, Fin.val_add, Fin.val_mul, Nat.pow_succ]
  simp only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod]

/-- The six generated residue terms are exactly the quadratic-nonresidue
packing of the centered cubic and centered-pair residual. -/
theorem generated_residue_polynomial (witness : Witness) :
    fieldResidue
        (evalPolynomial polynomialTerms
          (packedMatrixPoint 1 0 0 witness.residueLeft.val
            witness.residueRight.val)) =
      centeredResidual witness.residueLeft *
          centeredResidual witness.residueLeft +
        fieldResidue (goldilocksP - 7) *
          (witness.residuePairResidual * witness.residuePairResidual) := by
  rw [generated_residue_polynomial_expanded,
    centeredResidual_square_expansion,
    residuePairResidual_square_expansion,
    gateField_mul_add, gateField_mul_add]
  have middle :
      fieldResidue (goldilocksP - 7) *
          (fieldResidue (goldilocksP - 2) *
            (witness.residueLeft * witness.residueRight *
              witness.residueRight * witness.residueRight)) =
        fieldResidue 14 *
          (witness.residueLeft * witness.residueRight *
            witness.residueRight * witness.residueRight) := by
    rw [← gateField_mul_assoc, negSeven_mul_negTwo]
  rw [middle]
  have crossSeven :
      fieldResidue (goldilocksP - 7) *
          ((witness.residueLeft * witness.residueLeft) *
            (witness.residueRight * witness.residueRight)) =
        fieldResidue (goldilocksP - 7) *
          (witness.residueLeft * witness.residueLeft *
            witness.residueRight * witness.residueRight) := by
    ac_rfl
  have crossFourteen :
      fieldResidue 14 *
          (witness.residueLeft *
            (witness.residueRight * witness.residueRight *
              witness.residueRight)) =
        fieldResidue 14 *
          (witness.residueLeft * witness.residueRight *
            witness.residueRight * witness.residueRight) := by
    ac_rfl
  rw [crossSeven, crossFourteen]
  apply Fin.ext
  simp only [Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5
