import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame

/-!
Contract: locate every verifier-proof extension coordinate used by the
current 42-times-6 WASM benchmark in one stable physical namespace.

Assurance tier: model-level.

Owns: proof-coordinate numeric-location stability and the four Split-NC
endpoint families when only recursive relation matrix coefficients change.

Does not own: transcript input, emitted rows, semantic refinement, Rust, or
generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

private theorem kColumns_eq
    (left right : KColumns)
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp only at c0Equal c1Equal
  cases c0Equal
  cases c1Equal
  rfl

/-- A proof-coordinate numeric location is fixed by the complete namespace,
the proof bundle, and the two codec indices. Semantic value functions and
dependent membership proofs do not affect the location. -/
theorem proofNumeric_eq_of_ids_and_indices
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    {leftValue :
      (typeSystem leftParameters).Value (.data .nifsProof) →
        Nightstream.SuperNeo.Concrete.K}
    {rightValue :
      (typeSystem rightParameters).Value (.data .nifsProof) →
        Nightstream.SuperNeo.Concrete.K}
    (leftView :
      PaperNifsCodecProjection.KView
        (leftFamily.codecFor (.data .nifsProof)) leftValue)
    (rightView :
      PaperNifsCodecProjection.KView
        (rightFamily.codecFor (.data .nifsProof)) rightValue)
    (orderedEqual :
      PaperNifsGlobalColumnMap.orderedIds leftFrame =
        PaperNifsGlobalColumnMap.orderedIds rightFrame)
    (proofIdsEqual :
      (PaperNifsCallFrame.proofOperand leftFrame.operands).ids =
        (PaperNifsCallFrame.proofOperand rightFrame.operands).ids)
    (c0IndexEqual : leftView.c0Index.val = rightView.c0Index.val)
    (c1IndexEqual : leftView.c1Index.val = rightView.c1Index.val) :
    (PaperNifsGlobalColumnMap.kLocation leftFrame
      (leftView.columns
        (PaperNifsCallFrame.proofOperand leftFrame.operands)
        (PaperNifsCallFrame.proof_widthsAgree leftFrame))
      (PaperNifsGlobalColumnMap.proofOperand_mem leftFrame
        (leftView.c0_mem
          (PaperNifsCallFrame.proofOperand leftFrame.operands)
          (PaperNifsCallFrame.proof_widthsAgree leftFrame)))
      (PaperNifsGlobalColumnMap.proofOperand_mem leftFrame
        (leftView.c1_mem
          (PaperNifsCallFrame.proofOperand leftFrame.operands)
          (PaperNifsCallFrame.proof_widthsAgree leftFrame)))).numeric =
    (PaperNifsGlobalColumnMap.kLocation rightFrame
      (rightView.columns
        (PaperNifsCallFrame.proofOperand rightFrame.operands)
        (PaperNifsCallFrame.proof_widthsAgree rightFrame))
      (PaperNifsGlobalColumnMap.proofOperand_mem rightFrame
        (rightView.c0_mem
          (PaperNifsCallFrame.proofOperand rightFrame.operands)
          (PaperNifsCallFrame.proof_widthsAgree rightFrame)))
      (PaperNifsGlobalColumnMap.proofOperand_mem rightFrame
        (rightView.c1_mem
          (PaperNifsCallFrame.proofOperand rightFrame.operands)
          (PaperNifsCallFrame.proof_widthsAgree rightFrame)))).numeric := by
  unfold PaperNifsGlobalColumnMap.kLocation
  apply kColumns_eq
  · apply PaperNifsGlobalColumnMap.locate_source_congr
    · exact orderedEqual
    · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
        (PaperNifsCallFrame.proofOperand leftFrame.operands)
        (PaperNifsCallFrame.proofOperand rightFrame.operands)
        (PaperNifsCallFrame.proof_widthsAgree leftFrame)
        (PaperNifsCallFrame.proof_widthsAgree rightFrame)
        leftView.c0Index rightView.c0Index
        proofIdsEqual c0IndexEqual
  · apply PaperNifsGlobalColumnMap.locate_source_congr
    · exact orderedEqual
    · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
        (PaperNifsCallFrame.proofOperand leftFrame.operands)
        (PaperNifsCallFrame.proofOperand rightFrame.operands)
        (PaperNifsCallFrame.proof_widthsAgree leftFrame)
        (PaperNifsCallFrame.proof_widthsAgree rightFrame)
        leftView.c1Index rightView.c1Index
        proofIdsEqual c1IndexEqual

/-- Carried row expressions depend only on the two numeric locations.
Physical identity and membership proofs do not change the emitted terms. -/
theorem carried_eq_of_numeric_eq
    {leftMap rightMap : Nat → ColumnId}
    {leftTyped rightTyped : PaperNifsCodecProjection.KColumnIds}
    (left :
      PaperNifsCallColumnMap.KLocation leftMap leftTyped)
    (right :
      PaperNifsCallColumnMap.KLocation rightMap rightTyped)
    (numericEqual : left.numeric = right.numeric) :
    left.carried = right.carried := by
  unfold PaperNifsCallColumnMap.KLocation.carried
  rw [numericEqual]

/-- Singleton base-field expressions depend only on their numeric location. -/
theorem fCarried_eq_of_numeric_eq
    {leftMap rightMap : Nat → ColumnId}
    {leftTyped rightTyped : PaperNifsCodecProjection.FColumnId}
    (left :
      PaperNifsCallColumnMap.FLocation leftMap leftTyped)
    (right :
      PaperNifsCallColumnMap.FLocation rightMap rightTyped)
    (numericEqual : left.numeric = right.numeric) :
    left.carried = right.carried := by
  unfold PaperNifsCallColumnMap.FLocation.carried
  rw [numericEqual]

/-- A base-field proof coordinate is fixed by the complete namespace, the
proof bundle, and its codec index. The semantic value function and dependent
membership proofs do not affect the location. -/
theorem proofFNumeric_eq_of_ids_and_index
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    {leftValue :
      (typeSystem leftParameters).Value (.data .nifsProof) → Field}
    {rightValue :
      (typeSystem rightParameters).Value (.data .nifsProof) → Field}
    (leftView :
      PaperNifsCodecProjection.FView
        (leftFamily.codecFor (.data .nifsProof)) leftValue)
    (rightView :
      PaperNifsCodecProjection.FView
        (rightFamily.codecFor (.data .nifsProof)) rightValue)
    (orderedEqual :
      PaperNifsGlobalColumnMap.orderedIds leftFrame =
        PaperNifsGlobalColumnMap.orderedIds rightFrame)
    (proofIdsEqual :
      (PaperNifsCallFrame.proofOperand leftFrame.operands).ids =
        (PaperNifsCallFrame.proofOperand rightFrame.operands).ids)
    (indexEqual : leftView.index.val = rightView.index.val) :
    (PaperNifsGlobalColumnMap.fLocation leftFrame
      (leftView.column
        (PaperNifsCallFrame.proofOperand leftFrame.operands)
        (PaperNifsCallFrame.proof_widthsAgree leftFrame))
      (PaperNifsGlobalColumnMap.proofOperand_mem leftFrame
        (leftView.column_mem
          (PaperNifsCallFrame.proofOperand leftFrame.operands)
          (PaperNifsCallFrame.proof_widthsAgree leftFrame)))).numeric =
    (PaperNifsGlobalColumnMap.fLocation rightFrame
      (rightView.column
        (PaperNifsCallFrame.proofOperand rightFrame.operands)
        (PaperNifsCallFrame.proof_widthsAgree rightFrame))
      (PaperNifsGlobalColumnMap.proofOperand_mem rightFrame
        (rightView.column_mem
          (PaperNifsCallFrame.proofOperand rightFrame.operands)
          (PaperNifsCallFrame.proof_widthsAgree rightFrame)))).numeric := by
  unfold PaperNifsGlobalColumnMap.fLocation
  apply PaperNifsGlobalColumnMap.locate_source_congr
  · exact orderedEqual
  · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
      (PaperNifsCallFrame.proofOperand leftFrame.operands)
      (PaperNifsCallFrame.proofOperand rightFrame.operands)
      (PaperNifsCallFrame.proof_widthsAgree leftFrame)
      (PaperNifsCallFrame.proof_widthsAgree rightFrame)
      leftView.index rightView.index proofIdsEqual indexEqual

private theorem priorPoint_indices
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ coordinate,
      (((operational
        (template.withSystem left)).endpointViews.priorPoint
          coordinate).c0Index.val =
        ((operational
          (template.withSystem right)).endpointViews.priorPoint
            coordinate).c0Index.val) ∧
      (((operational
        (template.withSystem left)).endpointViews.priorPoint
          coordinate).c1Index.val =
        ((operational
          (template.withSystem right)).endpointViews.priorPoint
            coordinate).c1Index.val) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          intro coordinate
          constructor <;> rfl

private theorem claimedYRing_indices
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ running matrix lane,
      (((operational
        (template.withSystem left)).endpointViews.claimedYRing
          running matrix lane).c0Index.val =
        ((operational
          (template.withSystem right)).endpointViews.claimedYRing
            running matrix lane).c0Index.val) ∧
      (((operational
        (template.withSystem left)).endpointViews.claimedYRing
          running matrix lane).c1Index.val =
        ((operational
          (template.withSystem right)).endpointViews.claimedYRing
            running matrix lane).c1Index.val) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          intro running matrix lane
          constructor <;> rfl

private theorem outputYRing_indices
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ source matrix lane,
      (((operational
        (template.withSystem left)).endpointViews.outputYRing
          source matrix lane).c0Index.val =
        ((operational
          (template.withSystem right)).endpointViews.outputYRing
            source matrix lane).c0Index.val) ∧
      (((operational
        (template.withSystem left)).endpointViews.outputYRing
          source matrix lane).c1Index.val =
        ((operational
          (template.withSystem right)).endpointViews.outputYRing
            source matrix lane).c1Index.val) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          intro source matrix lane
          constructor <;> rfl

private theorem outputYZcol_indices
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ source lane,
      (((operational
        (template.withSystem left)).endpointViews.outputYZcol
          source lane).c0Index.val =
        ((operational
          (template.withSystem right)).endpointViews.outputYZcol
            source lane).c0Index.val) ∧
      (((operational
        (template.withSystem left)).endpointViews.outputYZcol
          source lane).c1Index.val =
        ((operational
          (template.withSystem right)).endpointViews.outputYZcol
            source lane).c1Index.val) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          intro source lane
          constructor <;> rfl

/-- Equal constraint polynomials give the same numeric location for every
prior-point endpoint coordinate. -/
theorem priorPoint_numeric_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ coordinate,
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem left)).family
        (invokePlan (template.withSystem left)).frame
        ((operational
          (template.withSystem left)).endpointViews.priorPoint
            coordinate)).numeric =
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem right)).family
        (invokePlan (template.withSystem right)).frame
        ((operational
          (template.withSystem right)).endpointViews.priorPoint
            coordinate)).numeric := by
  intro coordinate
  unfold ConcreteNifsOperationalFrame.proofLocation
  apply proofNumeric_eq_of_ids_and_indices
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact (priorPoint_indices template left right same coordinate).1
  · exact (priorPoint_indices template left right same coordinate).2

/-- Equal constraint polynomials give the same numeric location for every
claimed running-evaluation endpoint coordinate. -/
theorem claimedYRing_numeric_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ running matrix lane,
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem left)).family
        (invokePlan (template.withSystem left)).frame
        ((operational
          (template.withSystem left)).endpointViews.claimedYRing
            running matrix lane)).numeric =
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem right)).family
        (invokePlan (template.withSystem right)).frame
        ((operational
          (template.withSystem right)).endpointViews.claimedYRing
            running matrix lane)).numeric := by
  intro running matrix lane
  unfold ConcreteNifsOperationalFrame.proofLocation
  apply proofNumeric_eq_of_ids_and_indices
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact
      (claimedYRing_indices template left right same
        running matrix lane).1
  · exact
      (claimedYRing_indices template left right same
        running matrix lane).2

/-- Equal constraint polynomials give the same numeric location for every
output ring-evaluation endpoint coordinate. -/
theorem outputYRing_numeric_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ source matrix lane,
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem left)).family
        (invokePlan (template.withSystem left)).frame
        ((operational
          (template.withSystem left)).endpointViews.outputYRing
            source matrix lane)).numeric =
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem right)).family
        (invokePlan (template.withSystem right)).frame
        ((operational
          (template.withSystem right)).endpointViews.outputYRing
            source matrix lane)).numeric := by
  intro source matrix lane
  unfold ConcreteNifsOperationalFrame.proofLocation
  apply proofNumeric_eq_of_ids_and_indices
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact
      (outputYRing_indices template left right same source matrix lane).1
  · exact
      (outputYRing_indices template left right same source matrix lane).2

/-- Equal constraint polynomials give the same numeric location for every
output old-point endpoint coordinate. -/
theorem outputYZcol_numeric_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ source lane,
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem left)).family
        (invokePlan (template.withSystem left)).frame
        ((operational
          (template.withSystem left)).endpointViews.outputYZcol
            source lane)).numeric =
      (ConcreteNifsOperationalFrame.proofLocation
        (application (template.withSystem right)).family
        (invokePlan (template.withSystem right)).frame
        ((operational
          (template.withSystem right)).endpointViews.outputYZcol
            source lane)).numeric := by
  intro source lane
  unfold ConcreteNifsOperationalFrame.proofLocation
  apply proofNumeric_eq_of_ids_and_indices
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact (outputYZcol_indices template left right same source lane).1
  · exact (outputYZcol_indices template left right same source lane).2

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame
