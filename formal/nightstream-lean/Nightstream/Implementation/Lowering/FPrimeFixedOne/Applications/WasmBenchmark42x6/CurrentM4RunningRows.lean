import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningFrame

/-!
Contract: prove matrix-payload independence of the complete incoming
running-authority row family in the 42-times-6 WASM benchmark.

Assurance tier: model-level.

Owns: stability of commitment, public-input, evaluation, and point
recomposition coordinates; stability of their numeric rows; and stability
after translation into the physical call-frame namespace.

Does not own: other NIFS row families, activation, the recursive fixed point,
Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningRows

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private theorem fCoordinate_ext
    (left right : Phi81RadixRows.FCoordinate)
    (children : left.children = right.children)
    (parent : left.parent = right.parent) :
    left = right := by
  cases left
  cases right
  simp only at children parent
  cases children
  cases parent
  rfl

private theorem kCoordinate_ext
    (left right : Phi81RadixRows.KCoordinate)
    (children : left.children = right.children)
    (parent : left.parent = right.parent) :
    left = right := by
  cases left
  cases right
  simp only at children parent
  cases children
  cases parent
  rfl

private theorem carried_ext
    (left right : KMul.Carried)
    (low : left.low = right.low)
    (high : left.high = right.high) :
    left = right := by
  cases left
  cases right
  simp only at low high
  cases low
  cases high
  rfl

/-- Numeric running-authority rows before the physical call-frame map. -/
noncomputable def rows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsRunningAuthorityRows.rows
    (application setup) (operational setup) (invokePlan setup).frame

/-- Physical running-authority rows after the sole call-frame map. -/
noncomputable def physicalRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsRawProgram.translate
    (application setup) (invokePlan setup).frame (rows setup)

private theorem commitmentCoordinate_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (row : Fin verifierRows) (lane : Fin ringDegree) :
    ConcreteNifsRunningAuthorityRows.commitmentCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame row lane =
      ConcreteNifsRunningAuthorityRows.commitmentCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame row lane := by
  apply fCoordinate_ext
  · funext child
    change
      [(runningCoordinateNumeric (template.withSystem left)
          (.childCommitment child row lane), 1)] =
        [(runningCoordinateNumeric (template.withSystem right)
          (.childCommitment child row lane), 1)]
    rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
      template left right same]
  · change
      [(runningCoordinateNumeric (template.withSystem left)
          (.parentCommitment row lane), 1)] =
        [(runningCoordinateNumeric (template.withSystem right)
          (.parentCommitment row lane), 1)]
    rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
      template left right same]

private theorem publicCoordinate_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (column : Fin (ringDegree * publicRingColumns)) :
    ConcreteNifsRunningAuthorityRows.publicCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame column =
      ConcreteNifsRunningAuthorityRows.publicCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame column := by
  apply fCoordinate_ext
  · funext child
    change
      [(runningCoordinateNumeric (template.withSystem left)
          (.childPublic child column), 1)] =
        [(runningCoordinateNumeric (template.withSystem right)
          (.childPublic child column), 1)]
    rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
      template left right same]
  · change
      [(runningCoordinateNumeric (template.withSystem left)
          (.parentPublic column), 1)] =
        [(runningCoordinateNumeric (template.withSystem right)
          (.parentPublic column), 1)]
    rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
      template left right same]

private theorem evaluationCoordinate_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.shape.matrixCount) (lane : Fin ringDegree) :
    ConcreteNifsRunningAuthorityRows.evaluationCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame matrix lane =
      ConcreteNifsRunningAuthorityRows.evaluationCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame matrix lane := by
  apply kCoordinate_ext
  · funext child
    apply carried_ext
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.childEvaluation child matrix lane .c0), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.childEvaluation child matrix lane .c0), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.childEvaluation child matrix lane .c1), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.childEvaluation child matrix lane .c1), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]
  · apply carried_ext
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.parentEvaluation matrix lane .c0), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.parentEvaluation matrix lane .c0), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.parentEvaluation matrix lane .c1), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.parentEvaluation matrix lane .c1), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]

private theorem pointPair_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin dimensions.shape.rowVariables) :
    ConcreteNifsRunningAuthorityRows.pointPair
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame child coordinate =
      ConcreteNifsRunningAuthorityRows.pointPair
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame child coordinate := by
  apply Prod.ext
  · apply carried_ext
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.childPoint child coordinate .c0), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.childPoint child coordinate .c0), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.childPoint child coordinate .c1), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.childPoint child coordinate .c1), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]
  · apply carried_ext
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.parentPoint coordinate .c0), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.parentPoint coordinate .c0), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]
    · change
        [(runningCoordinateNumeric (template.withSystem left)
            (.parentPoint coordinate .c1), 1)] =
          [(runningCoordinateNumeric (template.withSystem right)
            (.parentPoint coordinate .c1), 1)]
      rw [runningCoordinateNumeric_eq_of_constraintPolynomial_eq
        template left right same]

private theorem pointPairs_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsRunningAuthorityRows.pointPairs
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsRunningAuthorityRows.pointPairs
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsRunningAuthorityRows.pointPairs
  apply congrArg
    (fun function =>
      (List.ofFn fun child : Fin productionGlobalParams.k => child).flatMap
        function)
  funext child
  apply congrArg List.ofFn
  funext coordinate
  exact pointPair_eq template left right same child coordinate

private theorem commitmentCoordinates_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsRunningAuthorityRows.commitmentCoordinates
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsRunningAuthorityRows.commitmentCoordinates
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsRunningAuthorityRows.commitmentCoordinates
  apply congrArg
    (fun function =>
      (List.ofFn fun row : Fin verifierRows => row).flatMap function)
  funext row
  apply congrArg List.ofFn
  funext lane
  exact commitmentCoordinate_eq template left right same row lane

private theorem publicCoordinates_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsRunningAuthorityRows.publicCoordinates
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsRunningAuthorityRows.publicCoordinates
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsRunningAuthorityRows.publicCoordinates
  apply congrArg List.ofFn
  funext column
  exact publicCoordinate_eq template left right same column

private theorem evaluationCoordinates_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsRunningAuthorityRows.evaluationCoordinates
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsRunningAuthorityRows.evaluationCoordinates
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsRunningAuthorityRows.evaluationCoordinates
  apply congrArg
    (fun function =>
      (List.ofFn fun matrix : Fin dimensions.shape.matrixCount => matrix
        ).flatMap function)
  funext matrix
  apply congrArg List.ofFn
  funext lane
  exact evaluationCoordinate_eq template left right same matrix lane

/-- Equal constraint polynomials give the same complete numeric
running-authority program. -/
theorem rows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    rows (template.withSystem left) =
      rows (template.withSystem right) := by
  unfold rows ConcreteNifsRunningAuthorityRows.rows
  unfold ConcreteNifsRunningAuthorityRows.fCoordinates
  rw [pointPairs_eq template left right same,
    commitmentCoordinates_eq template left right same,
    publicCoordinates_eq template left right same,
    evaluationCoordinates_eq template left right same]

/-- Equal constraint polynomials give the same physical translated
running-authority rows. -/
theorem physicalRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    physicalRows (template.withSystem left) =
      physicalRows (template.withSystem right) := by
  unfold physicalRows
  rw [rows_eq_of_constraintPolynomial_eq template left right same]
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold ConcreteNifsRawProgram.translate
          rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningRows
