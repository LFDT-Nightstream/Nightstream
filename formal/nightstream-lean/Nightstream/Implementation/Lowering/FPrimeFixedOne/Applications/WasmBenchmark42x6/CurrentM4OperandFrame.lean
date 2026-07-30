import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame

/-!
Contract: recover the exact fresh and proof operand bundles of the current
42-times-6 WASM benchmark from its stable complete NIFS namespace.

Assurance tier: model-level.

Owns: stability of the three operand bundle identities when only recursive
relation matrix coefficients change.

Does not own: codec projections, numeric locations, emitted rows, semantic
refinement, Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

private theorem operandIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (invokePlan (template.withSystem left)).frame.operands.ids =
      (invokePlan (template.withSystem right)).frame.operands.ids := by
  apply PaperNifsGlobalColumnMap.operand_ids_eq_of_orderedIds_eq
  · rw [PaperNifsCallFrame.operand_ids_length,
      PaperNifsCallFrame.operand_ids_length]
    cases left with
    | mk leftMatrices leftPolynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same

/-- Equal constraint polynomials give the same authoritative fresh operand
bundle. -/
theorem freshOperandIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    freshOperandIds (template.withSystem left) =
      freshOperandIds (template.withSystem right) := by
  let leftFrame := (invokePlan (template.withSystem left)).frame
  let rightFrame := (invokePlan (template.withSystem right)).frame
  apply PaperNifsGlobalColumnMap.segment_eq_of_joined_eq
    (PaperNifsCallFrame.runningOperand leftFrame.operands).ids
    (PaperNifsCallFrame.freshOperand leftFrame.operands).ids
    (PaperNifsCallFrame.proofOperand leftFrame.operands).ids
    (PaperNifsCallFrame.runningOperand rightFrame.operands).ids
    (PaperNifsCallFrame.freshOperand rightFrame.operands).ids
    (PaperNifsCallFrame.proofOperand rightFrame.operands).ids
  · exact congrArg List.length
      (runningOperandIds_eq_of_constraintPolynomial_eq
        template left right same)
  · have leftLength :=
      PaperNifsCallFrame.fresh_operand_ids_length leftFrame
    have rightLength :=
      PaperNifsCallFrame.fresh_operand_ids_length rightFrame
    refine Eq.trans leftLength (Eq.trans ?_ rightLength.symm)
    ·
      cases left with
      | mk leftMatrices leftPolynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl
  · calc
      (PaperNifsCallFrame.runningOperand leftFrame.operands).ids ++
            (PaperNifsCallFrame.freshOperand leftFrame.operands).ids ++
            (PaperNifsCallFrame.proofOperand leftFrame.operands).ids =
          leftFrame.operands.ids := by
            simpa only [List.append_assoc] using
              PaperNifsCallFrame.operand_ids leftFrame.operands
      _ = rightFrame.operands.ids :=
        operandIds_eq_of_constraintPolynomial_eq template left right same
      _ =
          (PaperNifsCallFrame.runningOperand rightFrame.operands).ids ++
            (PaperNifsCallFrame.freshOperand rightFrame.operands).ids ++
            (PaperNifsCallFrame.proofOperand rightFrame.operands).ids := by
              simpa only [List.append_assoc] using
                (PaperNifsCallFrame.operand_ids
                  rightFrame.operands).symm

/-- Equal constraint polynomials give the same authoritative proof operand
bundle. -/
theorem proofOperandIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    proofOperandIds (template.withSystem left) =
      proofOperandIds (template.withSystem right) := by
  let leftFrame := (invokePlan (template.withSystem left)).frame
  let rightFrame := (invokePlan (template.withSystem right)).frame
  have runningEqual :=
    runningOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  have freshEqual :=
    freshOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  apply PaperNifsGlobalColumnMap.segment_eq_of_joined_eq
    ((PaperNifsCallFrame.runningOperand leftFrame.operands).ids ++
      (PaperNifsCallFrame.freshOperand leftFrame.operands).ids)
    (PaperNifsCallFrame.proofOperand leftFrame.operands).ids
    []
    ((PaperNifsCallFrame.runningOperand rightFrame.operands).ids ++
      (PaperNifsCallFrame.freshOperand rightFrame.operands).ids)
    (PaperNifsCallFrame.proofOperand rightFrame.operands).ids
    []
  · exact congrArg List.length
      (congrArg₂ (· ++ ·) runningEqual freshEqual)
  · have leftLength :=
      PaperNifsCallFrame.proof_operand_ids_length leftFrame
    have rightLength :=
      PaperNifsCallFrame.proof_operand_ids_length rightFrame
    refine Eq.trans leftLength (Eq.trans ?_ rightLength.symm)
    ·
      cases left with
      | mk leftMatrices leftPolynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl
  · simpa only [List.append_assoc, List.append_nil] using
      (calc
        (PaperNifsCallFrame.runningOperand leftFrame.operands).ids ++
              (PaperNifsCallFrame.freshOperand leftFrame.operands).ids ++
              (PaperNifsCallFrame.proofOperand leftFrame.operands).ids =
            leftFrame.operands.ids := by
              simpa only [List.append_assoc] using
                PaperNifsCallFrame.operand_ids leftFrame.operands
        _ = rightFrame.operands.ids :=
          operandIds_eq_of_constraintPolynomial_eq template left right same
        _ =
            (PaperNifsCallFrame.runningOperand rightFrame.operands).ids ++
              (PaperNifsCallFrame.freshOperand rightFrame.operands).ids ++
              (PaperNifsCallFrame.proofOperand rightFrame.operands).ids := by
                simpa only [List.append_assoc] using
                (PaperNifsCallFrame.operand_ids
                  rightFrame.operands).symm)

/-- Equal constraint polynomials give the same authoritative running output
bundle. -/
theorem outputIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    outputIds (template.withSystem left) =
      outputIds (template.withSystem right) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold outputIds
          rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame
