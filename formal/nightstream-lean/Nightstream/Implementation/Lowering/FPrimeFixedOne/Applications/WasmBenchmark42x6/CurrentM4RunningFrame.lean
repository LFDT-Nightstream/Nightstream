import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame

/-!
Contract: erase setup-dependent proof fields from every physical coordinate
of the benchmark running carrier.

Assurance tier: model-level.

Owns: stability of every canonical running-carrier physical column and its
numeric location when only the recursive relation matrix payload changes.

Does not own: semantic equality of setup keys, NIFS row stability, the
recursive fixed point, Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningFrame

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- One semantic coordinate family in the selected benchmark running
carrier. -/
abbrev RunningCoordinate
    (dimensions : Dimensions) (verifierRows : Nat) :=
  ConcreteNifsCarrierViews.RunningCoordinate
    (ConcreteNifsPlain270Profile.Shape dimensions)
    publicRingColumns verifierRows

/-- Physical column selected by one running-coordinate descriptor. -/
noncomputable def runningCoordinateColumn
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows)
    (coordinate : RunningCoordinate dimensions verifierRows) : ColumnId :=
  let frame := (invokePlan setup).frame
  let view :=
    ConcreteNifsCarrierViews.RunningCoordinate.view
      (operational setup).runningViews coordinate
  (view.column
    (PaperNifsCallFrame.runningOperand frame.operands)
    (PaperNifsCallFrame.running_widthsAgree frame)).column

/-- Numeric source selected by one running-coordinate descriptor. -/
noncomputable def runningCoordinateNumeric
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows)
    (coordinate : RunningCoordinate dimensions verifierRows) : Nat :=
  let frame := (invokePlan setup).frame
  let view :=
    ConcreteNifsCarrierViews.RunningCoordinate.view
      (operational setup).runningViews coordinate
  (ConcreteNifsCarrierFrame.runningFLocation
    (application setup).family frame view).numeric

/-- Equal constraint polynomials select the same physical column for every
coordinate of the complete canonical running codec. -/
theorem runningCoordinateColumn_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ coordinate : RunningCoordinate dimensions verifierRows,
      runningCoordinateColumn (template.withSystem left) coordinate =
        runningCoordinateColumn (template.withSystem right) coordinate := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          intro coordinate
          cases coordinate <;> rfl

/-- Equal constraint polynomials select the same numeric source for every
coordinate of the complete canonical running codec. -/
theorem runningCoordinateNumeric_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ coordinate : RunningCoordinate dimensions verifierRows,
      runningCoordinateNumeric (template.withSystem left) coordinate =
        runningCoordinateNumeric (template.withSystem right) coordinate := by
  intro coordinate
  unfold runningCoordinateNumeric
  unfold ConcreteNifsCarrierFrame.runningFLocation
  unfold PaperNifsGlobalColumnMap.fLocation
  apply PaperNifsGlobalColumnMap.locate_source_congr
  · exact
      orderedIds_eq_of_constraintPolynomial_eq template left right same
  · exact
      runningCoordinateColumn_eq_of_constraintPolynomial_eq
        template left right same coordinate

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningFrame
