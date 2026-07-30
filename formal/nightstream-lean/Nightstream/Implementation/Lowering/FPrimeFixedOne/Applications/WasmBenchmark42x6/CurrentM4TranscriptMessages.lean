import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame

/-!
Contract: prove that every FE and NC prover-message coordinate used by the
current WASM benchmark has one stable physical numeric location.

Assurance tier: model-level.

Owns: coefficient-level and round-level stability for the selected Split-NC
transcript messages when only recursive relation matrix coefficients change.

Does not own: transcript assembly, endpoint authority, emitted rows, semantic
refinement, Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4TranscriptMessages

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

theorem feRowCoefficient_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀
      (round :
        Fin (ConcreteNifsPlain270Profile.Shape dimensions).rowVariables)
      (slot :
        Fin
          (SumCheck.Fe.Drow
            (KSplitNcStaticInput.layoutInput
              (operational (template.withSystem left)
                ).constraintPolynomial) + 1)),
      ConcreteNifsOperationalOccurrence.proofColumns
          (application (template.withSystem left)).family
          (invokePlan (template.withSystem left)).frame
          ((operational
            (template.withSystem left)).messageViews.feRow round slot) =
        ConcreteNifsOperationalOccurrence.proofColumns
          (application (template.withSystem right)).family
          (invokePlan (template.withSystem right)).frame
          ((operational
            (template.withSystem right)).messageViews.feRow round
              (Fin.cast
                (by
                  cases left with
                  | mk leftMatrices polynomial =>
                      cases right with
                      | mk rightMatrices rightPolynomial =>
                          simp only at same
                          subst rightPolynomial
                          rfl)
                slot)) := by
  intro round slot
  unfold ConcreteNifsOperationalOccurrence.proofColumns
  unfold ConcreteNifsOperationalFrame.proofLocation
  apply proofNumeric_eq_of_ids_and_indices
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  · cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

theorem feLaneCoefficient_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀
      (round :
        Fin
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.laneVariables)
      (slot : Fin 3),
      ConcreteNifsOperationalOccurrence.proofColumns
          (application (template.withSystem left)).family
          (invokePlan (template.withSystem left)).frame
          ((operational
            (template.withSystem left)).messageViews.feLane round slot) =
        ConcreteNifsOperationalOccurrence.proofColumns
          (application (template.withSystem right)).family
          (invokePlan (template.withSystem right)).frame
          ((operational
            (template.withSystem right)).messageViews.feLane round slot) := by
  intro round slot
  unfold ConcreteNifsOperationalOccurrence.proofColumns
  unfold ConcreteNifsOperationalFrame.proofLocation
  apply proofNumeric_eq_of_ids_and_indices
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  · cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

theorem ncCoefficient_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀
      (round :
        Fin
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.blockVariables +
            Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.laneVariables))
      (slot : Fin 5),
      ConcreteNifsOperationalOccurrence.proofColumns
          (application (template.withSystem left)).family
          (invokePlan (template.withSystem left)).frame
          ((operational
            (template.withSystem left)).messageViews.nc round slot) =
        ConcreteNifsOperationalOccurrence.proofColumns
          (application (template.withSystem right)).family
          (invokePlan (template.withSystem right)).frame
          ((operational
            (template.withSystem right)).messageViews.nc round slot) := by
  intro round slot
  unfold ConcreteNifsOperationalOccurrence.proofColumns
  unfold ConcreteNifsOperationalFrame.proofLocation
  apply proofNumeric_eq_of_ids_and_indices
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  · cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4TranscriptMessages
