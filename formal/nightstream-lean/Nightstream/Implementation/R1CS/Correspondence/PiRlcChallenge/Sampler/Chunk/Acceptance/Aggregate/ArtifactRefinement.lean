import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptance
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.ArtifactEvaluation

/-!
Artifact-checked refinement of the generated aggregate-acceptance leaf.

Owns: exact generated shape, matrix binding and polynomial checks; direct
interpretation of each of the nine generated rows; and their equivalence to
the independently defined `AggregateAcceptanceRows` relation.

Does not own: source-bit decoder placement, selectors, inactive rows, the
recursive 960-chunk physical image, cost totals, security reduction, or row
removal authority.

Emits constraints: no.

Authority boundary: generated data is non-authoritative evidence. The final
equivalence is stated against independent source/tree semantics. It remains
leaf-local until a physical outer-image theorem supplies the role assignment.

| Stage path | Generated evidence | Mathematical obligation | Assurance tier |
|---|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.tree_bit_pairs` | seven normalized rows | paired Boolean residual equations | artifact-checked, leaf-local |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.product_aggregate` | one normalized row | radix-three product equality | artifact-checked, leaf-local |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.root_binding` | one normalized row | accept equals one minus both roots | artifact-checked, leaf-local |
| gate specialization | arity 56, 40 bindings, 25 terms | exact occupied production polynomial | artifact-checked, leaf-local |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance

open Nightstream.Implementation.R1CS
open Mod5
open AggregateAcceptanceArtifact
open AggregateAcceptanceArtifactData

/-- Exact finite shape of the regenerated active payload. -/
theorem generated_aggregate_shape_exact :
    schemaVersion = 2 ∧
      gateArity = 56 ∧
      matrixBindings.length = 40 ∧
      activeRows.length = 9 ∧
      polynomialTerms.length = 25 := by
  native_decide

/-- Exact production matrix index for each finite aggregate role. -/
def AggregateAcceptanceArtifact.MatrixRole.index : MatrixRole → Nat
  | .selector => 0
  | .productLeft slot => 3 + slot
  | .productRight slot => 21 + slot
  | .productOut => 39
  | .quadraticBitLeft => 44
  | .quadraticBitRight => 45

theorem generated_aggregate_matrix_bindings_exact :
    ∀ binding ∈ matrixBindings,
      binding.index = binding.role.index := by
  native_decide

/-- Exact selector-inclusive degree sequence of the generated specialization. -/
theorem generated_aggregate_polynomial_degrees_exact :
    polynomialTerms.map PolynomialTerm.totalDegree =
      [3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3,
       2, 5, 4, 3, 5, 4, 3] := by
  native_decide

private theorem coefficient_one : coefficient 1 = fieldResidue 1 := by
  rfl

private theorem coefficient_negOne :
    coefficient (-1) = fieldResidue (goldilocksP - 1) := by
  native_decide

private theorem coefficient_negTwo :
    coefficient (-2) = fieldResidue (goldilocksP - 2) := by
  native_decide

private theorem coefficient_negSeven :
    coefficient (-7) = fieldResidue (goldilocksP - 7) := by
  native_decide

private theorem coefficient_fourteen : coefficient 14 = fieldResidue 14 := by
  rfl

private theorem coefficient_three : coefficient 3 = fieldResidue 3 := by rfl
private theorem coefficient_nine : coefficient 9 = fieldResidue 9 := by rfl
private theorem coefficient_twentySeven : coefficient 27 = fieldResidue 27 := by rfl
private theorem coefficient_eightyOne : coefficient 81 = fieldResidue 81 := by rfl
private theorem coefficient_243 : coefficient 243 = fieldResidue 243 := by rfl
private theorem coefficient_729 : coefficient 729 = fieldResidue 729 := by rfl
private theorem coefficient_2187 : coefficient 2187 = fieldResidue 2187 := by rfl
private theorem coefficient_6561 : coefficient 6561 = fieldResidue 6561 := by rfl
private theorem coefficient_19683 : coefficient 19683 = fieldResidue 19683 := by rfl
private theorem coefficient_59049 : coefficient 59049 = fieldResidue 59049 := by rfl
private theorem coefficient_177147 : coefficient 177147 = fieldResidue 177147 := by rfl
private theorem coefficient_531441 : coefficient 531441 = fieldResidue 531441 := by rfl
private theorem coefficient_1594323 : coefficient 1594323 = fieldResidue 1594323 := by rfl

private theorem generatedBitRow_polynomial
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) (row : ActiveRow) (left right : Fin 14)
    (rowExact : row =
      [⟨.selector, [⟨.one, 1⟩]⟩,
       ⟨.quadraticBitLeft, [⟨.treeOutput left, 1⟩]⟩,
       ⟨.quadraticBitRight, [⟨.treeOutput right, 1⟩]⟩]) :
    evalPolynomial polynomialTerms
        (row.point (coordinateAssignment bits outputs accept)) =
      booleanPairGatePolynomial (outputs left) (outputs right) := by
  rw [rowExact]
  simp [evalPolynomial, evalPolynomialTerm, evalPowers, polynomialTerms,
    ActiveRow.point, MatrixLinearCombination.value,
    evalLinearCombination, coordinateAssignment, coefficient_one,
    coefficient_negOne, coefficient_negTwo, coefficient_negSeven,
    coefficient_fourteen, fieldResidue_one, gateField_one_mul,
    gateField_mul_one, gateField_mul_zero]
  unfold booleanPairGatePolynomial
  simp only [gateField_add_assoc]

private theorem generatedBitRow_iff
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) (row : ActiveRow) (left right : Fin 14)
    (rowExact : row =
      [⟨.selector, [⟨.one, 1⟩]⟩,
       ⟨.quadraticBitLeft, [⟨.treeOutput left, 1⟩]⟩,
       ⟨.quadraticBitRight, [⟨.treeOutput right, 1⟩]⟩]) :
    row.Holds polynomialTerms (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (bitResidual (outputs left))
        (bitResidual (outputs right)) := by
  rw [ActiveRow.Holds,
    generatedBitRow_polynomial bits outputs accept row left right rowExact,
    booleanPairGatePolynomial_eq_quadratic]
  rfl

/-- The seven generated Boolean-pair rows, separated from the aggregate and
root-binding rows so the artifact tree mirrors the semantic tree. -/
def GeneratedProductTreeOutputBitRows
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) : Prop :=
  (activeRows[0]).Holds polynomialTerms
      (coordinateAssignment bits outputs accept) ∧
    (activeRows[1]).Holds polynomialTerms
      (coordinateAssignment bits outputs accept) ∧
    (activeRows[2]).Holds polynomialTerms
      (coordinateAssignment bits outputs accept) ∧
    (activeRows[3]).Holds polynomialTerms
      (coordinateAssignment bits outputs accept) ∧
    (activeRows[4]).Holds polynomialTerms
      (coordinateAssignment bits outputs accept) ∧
    (activeRows[5]).Holds polynomialTerms
      (coordinateAssignment bits outputs accept) ∧
    (activeRows[6]).Holds polynomialTerms
      (coordinateAssignment bits outputs accept)

theorem generatedProductTreeOutputBitRows_iff
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    GeneratedProductTreeOutputBitRows bits outputs accept ↔
      ProductTreeOutputBitRows outputs := by
  unfold GeneratedProductTreeOutputBitRows ProductTreeOutputBitRows
  rw [generatedBitRow_iff bits outputs accept (activeRows[0]) 0 1 (by rfl),
    generatedBitRow_iff bits outputs accept (activeRows[1]) 2 3 (by rfl),
    generatedBitRow_iff bits outputs accept (activeRows[2]) 4 5 (by rfl),
    generatedBitRow_iff bits outputs accept (activeRows[3]) 6 7 (by rfl),
    generatedBitRow_iff bits outputs accept (activeRows[4]) 8 9 (by rfl),
    generatedBitRow_iff bits outputs accept (activeRows[5]) 10 11 (by rfl),
    generatedBitRow_iff bits outputs accept (activeRows[6]) 12 13 (by rfl)]

private theorem generatedProductAggregateRow_polynomial
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    evalPolynomial polynomialTerms
        ((activeRows[7]).point (coordinateAssignment bits outputs accept)) =
      fieldSub
        (radix3Field (productTreeProducts bits outputs))
        (radix3Field outputs) := by
  simp [evalPolynomial, evalPolynomialTerm, evalPowers, polynomialTerms,
    activeRows, ActiveRow.point, MatrixLinearCombination.value,
    evalLinearCombination, coordinateAssignment, coefficient_one,
    coefficient_negOne, coefficient_negTwo, coefficient_negSeven,
    coefficient_fourteen, fieldResidue_one, gateField_one_mul,
    gateField_mul_one, gateField_mul_zero, productTreeProducts,
    productTreeLeft, productTreeRight, fieldSub, radix3Field,
    gateField_mul_add, gateField_mul_assoc,
    fieldResidue_mul_residue_mul, coefficient_three, coefficient_nine,
    coefficient_twentySeven, coefficient_eightyOne, coefficient_243,
    coefficient_729, coefficient_2187, coefficient_6561,
    coefficient_19683, coefficient_59049, coefficient_177147,
    coefficient_531441, coefficient_1594323]
  simp only [gateField_add_assoc]

/-- The generated radix-three product row as a separate artifact family. -/
def GeneratedProductTreeAggregateRow
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) : Prop :=
  (activeRows[7]).Holds polynomialTerms
    (coordinateAssignment bits outputs accept)

theorem generatedProductTreeAggregateRow_iff
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    GeneratedProductTreeAggregateRow bits outputs accept ↔
      ProductTreeAggregateRow bits outputs := by
  unfold GeneratedProductTreeAggregateRow ActiveRow.Holds
  rw [generatedProductAggregateRow_polynomial, fieldSub_eq_zero_iff,
    radix3Field_eq_residue, radix3Field_eq_residue,
    fieldResidue_eq_iff_mod]
  unfold ProductTreeAggregateRow
  constructor <;> exact Eq.symm

private theorem generatedFinalAcceptanceRow_polynomial
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    evalPolynomial polynomialTerms
        ((activeRows[8]).point (coordinateAssignment bits outputs accept)) =
      fieldSub accept (fieldSub 1 (outputs 6 * outputs 13)) := by
  simp [evalPolynomial, evalPolynomialTerm, evalPowers, polynomialTerms,
    activeRows, ActiveRow.point, MatrixLinearCombination.value,
    evalLinearCombination, coordinateAssignment, coefficient_one,
    coefficient_negOne, coefficient_negTwo, coefficient_negSeven,
    coefficient_fourteen, fieldResidue_one, gateField_one_mul,
    gateField_mul_one, gateField_mul_zero, fieldSub,
    gateField_mul_add, negOne_mul_negOne_mul]
  apply gateField_add_outer_comm

/-- The generated root/accept row as a separate artifact family. -/
def GeneratedFinalAcceptanceRow
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) : Prop :=
  (activeRows[8]).Holds polynomialTerms
    (coordinateAssignment bits outputs accept)

theorem generatedFinalAcceptanceRow_iff
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    GeneratedFinalAcceptanceRow bits outputs accept ↔
      FinalAcceptanceRow outputs accept := by
  unfold GeneratedFinalAcceptanceRow ActiveRow.Holds FinalAcceptanceRow
  rw [generatedFinalAcceptanceRow_polynomial, fieldSub_eq_zero_iff]

/-- Exact generated nine-row leaf, grouped by the same three mathematical
families as the independent relation. -/
def GeneratedAggregateAcceptanceRows
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) : Prop :=
  GeneratedProductTreeOutputBitRows bits outputs accept ∧
    GeneratedProductTreeAggregateRow bits outputs accept ∧
    GeneratedFinalAcceptanceRow bits outputs accept

/-- The active arity-56 generated leaf is exactly the independent nine-row
aggregate acceptance relation. This remains leaf-local: it says nothing about
the recursive 960-chunk physical image. -/
theorem generatedAggregateAcceptanceRows_iff
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    GeneratedAggregateAcceptanceRows bits outputs accept ↔
      AggregateAcceptanceRows bits outputs accept := by
  unfold GeneratedAggregateAcceptanceRows AggregateAcceptanceRows
  rw [generatedProductTreeOutputBitRows_iff,
    generatedProductTreeAggregateRow_iff,
    generatedFinalAcceptanceRow_iff]

theorem generatedAggregateAcceptanceRows_iff_sourceMeaning
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) (sourceBoolean : ∀ index, FieldBit (bits index)) :
    GeneratedAggregateAcceptanceRows bits outputs accept ↔
      ProductTreeMeaning bits outputs ∧
        SourceAcceptanceMeaning bits accept := by
  rw [generatedAggregateAcceptanceRows_iff,
    aggregateAcceptanceRows_iff_sourceMeaning prime nonresidue
      bits outputs accept sourceBoolean]

theorem generatedAggregateAcceptanceRows_iff_verifierMeaning
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : BitsBoolean assignment chunk) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    GeneratedAggregateAcceptanceRows
        (sourceBits assignment chunk) outputs accept ↔
      ProductTreeMeaning (sourceBits assignment chunk) outputs ∧
        VerifierAcceptanceMeaning assignment chunk bits accept := by
  rw [generatedAggregateAcceptanceRows_iff,
    aggregateAcceptanceRows_iff_verifierMeaning prime nonresidue
      bits outputs accept]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance
