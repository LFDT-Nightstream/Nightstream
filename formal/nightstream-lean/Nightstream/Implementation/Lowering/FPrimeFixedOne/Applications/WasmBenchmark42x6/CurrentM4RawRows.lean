import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ActionRows

/-!
Contract: prove physical row stability for the complete raw Step receipt list.

Assurance tier: model-level.

Owns: exact equality of every raw physical Step row when relation matrices
change but the selected SuperNeo constraint polynomial does not.

Does not own: column identity stability, fixed-point compilation, production
selection, Rust equality, or a security reduction.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RawRows

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInput
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInputEquality
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ActionRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4NifsRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

theorem rawRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsRawProgram.rawRows
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsRawProgram.rawRows
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  have proofEqual :=
    CurrentM4ProofRows.rows_eq_of_constraintPolynomial_eq
      template left right same
  unfold CurrentM4ProofRows.rows at proofEqual
  have runningEqual :=
    CurrentM4RunningRows.physicalRows_eq_of_constraintPolynomial_eq
      template left right same
  unfold CurrentM4RunningRows.physicalRows
    CurrentM4RunningRows.rows at runningEqual
  have samplerSourceEqual :=
    samplerRows_eq_of_constraintPolynomial_eq
      template left right same
  unfold samplerRows at samplerSourceEqual
  have samplerEqual :=
    translate_eq_of_constraintPolynomial_eq
      template left right same _ _ samplerSourceEqual
  have pointSourceEqual :=
    pointRows_eq_of_constraintPolynomial_eq
      template left right same
  unfold pointRows at pointSourceEqual
  have pointEqual :=
    translate_eq_of_constraintPolynomial_eq
      template left right same _ _ pointSourceEqual
  have actionEqual :=
    actionRows_eq_of_constraintPolynomial_eq
      template left right same
  unfold actionRows at actionEqual
  have piDecSourceEqual :=
    piDecRows_eq_of_constraintPolynomial_eq
      template left right same
  unfold piDecRows at piDecSourceEqual
  have piDecEqual :=
    translate_eq_of_constraintPolynomial_eq
      template left right same _ _ piDecSourceEqual
  have outputSourceEqual :=
    outputRows_eq_of_constraintPolynomial_eq
      template left right same
  unfold outputRows at outputSourceEqual
  have outputEqual :=
    translate_eq_of_constraintPolynomial_eq
      template left right same _ _ outputSourceEqual
  unfold ConcreteNifsRawProgram.rawRows
  rw [proofEqual, runningEqual, samplerEqual, pointEqual,
    actionEqual, piDecEqual, outputEqual]

theorem allocationWidth_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsRawProgram.allocationWidth
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsRawProgram.allocationWidth
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          have inputEqual :
              input
                  (template.withSystem
                    { matrices := leftMatrices
                      constraintPolynomial := polynomial }) =
                input
                  (template.withSystem
                    { matrices := rightMatrices
                      constraintPolynomial := polynomial }) :=
            eq_of_heq
              (input_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl)
          unfold ConcreteNifsRawProgram.allocationWidth
            ConcreteNifsOperationalSampler.cost
            ConcreteNifsOperationalSampler.challengeCost
          simp only [
            Nightstream.Implementation.Lowering.Typed.Cost.add_auxiliaryColumns]
          rw [show
            ConcreteNifsOperationalOccurrence.input
                (application
                  (template.withSystem
                    { matrices := leftMatrices
                      constraintPolynomial := polynomial }))
                (operational
                  (template.withSystem
                    { matrices := leftMatrices
                      constraintPolynomial := polynomial }))
                (invokePlan
                  (template.withSystem
                    { matrices := leftMatrices
                      constraintPolynomial := polynomial })).frame =
              ConcreteNifsOperationalOccurrence.input
                (application
                  (template.withSystem
                    { matrices := rightMatrices
                      constraintPolynomial := polynomial }))
                (operational
                  (template.withSystem
                    { matrices := rightMatrices
                      constraintPolynomial := polynomial }))
                (invokePlan
                  (template.withSystem
                    { matrices := rightMatrices
                      constraintPolynomial := polynomial })).frame
            from inputEqual]
          rfl

theorem residuals_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsActivatedProgram.residuals
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsActivatedProgram.residuals
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsActivatedProgram.residuals
  exact congrArg₂ List.drop
    (allocationWidth_eq_of_constraintPolynomial_eq
      template left right same)
    (temporaryIds_eq_of_constraintPolynomial_eq
      template left right same)

theorem activatedRawRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsActivatedProgram.rawRows
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsActivatedProgram.rawRows
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsActivatedProgram.rawRows
  rw [active_eq_of_constraintPolynomial_eq
      template left right same,
    rawRows_eq_of_constraintPolynomial_eq
      template left right same,
    residuals_eq_of_constraintPolynomial_eq
      template left right same]

theorem activatedRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsActivatedProgram.rows
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsActivatedProgram.rows
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsActivatedProgram.rows
  rw [owner_eq_of_constraintPolynomial_eq
      template left right same,
    active_eq_of_constraintPolynomial_eq
      template left right same,
    rawRows_eq_of_constraintPolynomial_eq
      template left right same,
    residuals_eq_of_constraintPolynomial_eq
      template left right same]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RawRows
